using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for graph/mesh layers that require setup before Forward
/// (adjacency matrices, Laplacians, eigenbases, spiral indices, etc.).
/// Subclasses override SetupLayer() to provide domain-specific initialization.
/// </summary>
public abstract class GraphLayerTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected static T ToT(double value) => NumOps.FromDouble(value);
    protected static double ToD(T value) => Convert.ToDouble(value);
    protected virtual double Tolerance => typeof(T) == typeof(float) ? 1e-6 : 1e-12;

    protected abstract ILayer<T> CreateLayer();

    /// <summary>
    /// Perform domain-specific setup on the layer (set adjacency matrix, Laplacian, etc.).
    /// Called before every test's Forward pass.
    /// </summary>
    protected abstract void SetupLayer(ILayer<T> layer);

    /// <summary>Shape of the input tensor. Default: [4, 8] (4 nodes, 8 features).</summary>
    protected virtual int[] InputShape => [4, 8];

    /// <summary>Whether the layer has trainable parameters. Default: true.</summary>
    protected virtual bool ExpectsTrainableParameters => true;

    protected static Tensor<T> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = ToT(rng.NextDouble() * 4.0 + 1.0); // [1.0, 5.0] — large positive to survive ReLU after matmul
        return tensor;
    }

    private ILayer<T> CreateAndSetup()
    {
        var layer = CreateLayer();
        SetupLayer(layer);
        return layer;
    }

    // =========================================================================
    // INVARIANT 1: Forward produces finite output
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateAndSetup();
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);

        Assert.True(output.Length > 0, "Output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(ToD(output[i])), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(ToD(output[i])), $"Output[{i}] is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 2: Forward is deterministic
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateAndSetup();
        layer.SetTrainingMode(false);
        var input = CreateRandomTensor(InputShape);

        var out1 = layer.Forward(input);
        layer.ResetState();
        var out2 = layer.Forward(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }

    // =========================================================================
    // INVARIANT 3: Different inputs produce different outputs
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateAndSetup();
        layer.SetTrainingMode(false);

        // Mixed-sign inputs, not CreateRandomTensor's all-positive [1, 5]. Weights are drawn from the shared
        // thread-local RNG, so a layer's initialization depends on which tests ran on the thread before it. With
        // all-positive inputs a ReLU layer whose random weights make every output channel's pre-activation
        // negative outputs all zeros for BOTH inputs - probability 2^-C, 1 in 16 for DiffusionConvLayer's four
        // channels - and this invariant failed in the full suite while passing alone. Mixed signs make every
        // vertex/channel pre-activation vary in sign, so both outputs all dead has negligible probability.
        var input1 = CreateMixedSignTensor(InputShape, seed: 1);
        var input2 = CreateMixedSignTensor(InputShape, seed: 2);

        layer.ResetState();
        var output1 = layer.Forward(input1);
        layer.ResetState();
        var output2 = layer.Forward(input2);

        bool anyDifferent = false;
        double maxAbs1 = 0, maxAbs2 = 0;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            maxAbs1 = Math.Max(maxAbs1, Math.Abs(ToD(output1[i])));
            maxAbs2 = Math.Max(maxAbs2, Math.Abs(ToD(output2[i])));
            if (Math.Abs(ToD(output1[i]) - ToD(output2[i])) > Tolerance)
                anyDifferent = true;
        }
        Assert.True(anyDifferent,
            $"Layer produces identical output for different inputs (max |output| {maxAbs1:G6} and {maxAbs2:G6} " +
            $"over {minLen} values; both zero means every activation is dead for these weights).");
    }

    private static Tensor<T> CreateMixedSignTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = ToT(rng.NextDouble() * 4.0 - 2.0); // [-2, 2]
        return tensor;
    }

    // =========================================================================
    // INVARIANT 4: Backward produces finite gradient
    // =========================================================================


    // =========================================================================
    // INVARIANT 5: Parameter count consistency
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Parameters_CountShouldMatchVector()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateAndSetup();

        // Drive lazy-shape resolution + weight allocation by running a
        // probe Forward against InputShape. Without this, lazy graph
        // layers (SpiralConvLayer, GraphAttention, etc.) report
        // ParameterCount = 0 because their weights resolve only on the
        // first Forward call. CreateAndSetup may already have called
        // SetupGraph or similar, but the layer's weights are still
        // allocated lazily.
        try
        {
            using var probe = CreateRandomTensor(InputShape);
            layer.Forward(probe);
        }
        catch
        {
            // Some layers reject the default InputShape — the invariant
            // still validates whatever state the ctor produced.
        }

        int count = (int)layer.ParameterCount;
        var parameters = layer.GetParameters();

        Assert.True(count >= 0, "ParameterCount should be non-negative.");
        Assert.Equal(count, parameters.Length);

        if (ExpectsTrainableParameters)
            Assert.True(count > 0, "Layer should have trainable parameters.");
    }

    // =========================================================================
    // INVARIANT 6: ResetState doesn't break the layer
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task ResetState_ShouldNotBreakForward()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateAndSetup();
        var input = CreateRandomTensor(InputShape);

        layer.Forward(input);
        layer.ResetState();

        var output = layer.Forward(input);
        Assert.True(output.Length > 0);
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(ToD(output[i])), $"Output[{i}] is NaN after ResetState.");
    }
}

/// <summary>Default-precision alias for existing hand-written fixtures.</summary>
public abstract class GraphLayerTestBase : GraphLayerTestBase<double> { }
