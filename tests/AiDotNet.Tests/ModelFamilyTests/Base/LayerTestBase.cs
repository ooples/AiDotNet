using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Loss strategies for gradient checking. Each produces a different gradient signal
/// to expose different classes of backward pass bugs.
/// </summary>
public enum GradientCheckLossStrategy
{
    /// <summary>L = sum(x^2)/2, dL/dx = x. Gradient proportional to output — bugs where backward
    /// multiplies by output direction cancel out and become invisible.</summary>
    MSE,

    /// <summary>L = sum(w*x) with fixed random w, dL/dx = w. Random gradient direction with no
    /// alignment to output — exposes bugs hidden by MSE's output-aligned gradient.</summary>
    RandomProjection,

    /// <summary>L = sum(huber(x)), dL/dx = x for |x|&lt;1, sign(x) for |x|&gt;=1. Smooth L1 — constant
    /// magnitude gradients for large values, differentiable everywhere (unlike raw L1 which causes
    /// false positives at x=0 in finite difference checks).</summary>
    Huber,
}

/// <summary>
/// Base test class for ILayer&lt;double&gt; implementations.
/// Tests mathematical invariants that every layer must satisfy:
/// finite forward output, backward gradient flow, parameter consistency,
/// serialization roundtrip, input sensitivity, and gradient correctness.
///
/// Subclasses override CreateLayer() and optionally InputShape/OutputShape.
/// All invariant tests are inherited automatically.
///
/// Gradient checking uses multiple loss strategies (MSE, RandomProjection, Huber) to expose
/// bugs hidden by specific gradient alignments. Activation functions are auto-discovered
/// via reflection so new activations are automatically tested.
/// </summary>
public abstract class LayerTestBase
{
    /// <summary>
    /// Factory method — create a fresh instance of the layer under test.
    /// </summary>
    protected abstract ILayer<double> CreateLayer();

    /// <summary>
    /// Shape of the tensor to feed into Forward. Override for layers that need
    /// specific shapes (e.g. [batch, channels, height, width] for conv layers).
    /// Default: [1, 4] — single sample, 4 features.
    /// </summary>
    protected virtual int[] InputShape => [1, 4];

    /// <summary>
    /// Whether the layer is expected to have trainable parameters.
    /// Override to false for pass-through layers (InputLayer, FlattenLayer, ActivationLayer, etc.)
    /// </summary>
    protected virtual bool ExpectsTrainableParameters => true;

    /// <summary>
    /// Whether the layer's gradient computation produces meaningful gradients.
    /// Some layers (ReservoirLayer, InputLayer) pass gradients through but
    /// don't compute weight gradients. Override to false for those.
    /// Note: With tape-based autodiff, gradient correctness is verified through
    /// the tape rather than layer-level Backward.
    /// </summary>
    protected virtual bool ExpectsNonZeroGradients => true;

    /// <summary>
    /// Tolerance for numerical comparisons. Layers with stochastic behavior
    /// (dropout, noise) may need higher tolerance.
    /// </summary>
    protected virtual double Tolerance => 1e-12;

    /// <summary>
    /// Loss strategy for the basic gradient check (Invariant 12).
    /// Capsule layers should use RandomProjection because MSE gradient aligns with Squash
    /// output direction, which the Squash Jacobian attenuates.
    /// Note: The loss variant Theory test (Invariant 13) tests ALL strategies regardless.
    /// </summary>
    protected virtual GradientCheckLossStrategy DefaultLossStrategy => GradientCheckLossStrategy.MSE;

    /// <summary>
    /// Whether constant inputs (all 0.1 vs all 0.9) should produce different outputs.
    /// False for normalization layers (LayerNorm, BatchNorm on single-feature constant input)
    /// where constant inputs normalize to the same output by design.
    /// </summary>
    protected virtual bool ExpectsDifferentOutputForConstantInputs => true;

    /// <summary>
    /// Whether this layer supports testing with different activation functions.
    /// Override to true and implement CreateLayerWithActivation() for layers that
    /// accept activation function parameters in their options/constructor.
    /// When true, the ActivationVariant Theory test runs with every auto-discovered activation.
    /// </summary>
    protected virtual bool SupportsActivationVariants => false;

    /// <summary>
    /// Creates the layer under test with a specific activation function injected.
    /// Override for layers that accept activation parameters in their options.
    /// Default: returns CreateLayer() (ignoring the activation parameter).
    /// </summary>
    protected virtual ILayer<double> CreateLayerWithActivation(ActivationFunctionBase<double> activation)
        => CreateLayer();


    // =========================================================================
    // Static discovery infrastructure
    // Auto-discovers activation functions via reflection so that adding a new
    // ActivationFunctionBase<T> implementation automatically includes it in tests.
    // =========================================================================

    private static readonly Lazy<IReadOnlyList<(string Name, Type ClosedType)>> _activationCache =
        new(DiscoverScalarActivationTypes);

    /// <summary>
    /// Discovers all concrete ActivationFunctionBase&lt;double&gt; implementations that support
    /// scalar operations. Vector-only activations (Squash, Softmax, etc.) are excluded.
    /// Results are cached — discovery only runs once per test session.
    /// </summary>
    private static IReadOnlyList<(string Name, Type ClosedType)> DiscoverScalarActivationTypes()
    {
        // Force-load the AiDotNet assembly so its types are discoverable
        _ = typeof(ActivationFunctionBase<>).Assembly;

        var openBase = typeof(ActivationFunctionBase<>);
        var results = new List<(string, Type)>();

        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type[] types;
            try { types = assembly.GetTypes(); }
            catch (ReflectionTypeLoadException ex)
            {
                types = ex.Types.Where(t => t is not null).ToArray()!;
            }
            catch { continue; }

            foreach (var type in types)
            {
                if (type.IsAbstract || type.IsInterface || !type.IsGenericTypeDefinition)
                    continue;

                // Walk inheritance chain to check for ActivationFunctionBase<>
                if (!InheritsFromOpenGeneric(type, openBase))
                    continue;

                try
                {
                    var closedType = type.MakeGenericType(typeof(double));
                    if (Activator.CreateInstance(closedType) is not ActivationFunctionBase<double> instance)
                        continue;

                    // Test scalar support by trying Activate — vector-only activations throw
                    try
                    {
                        instance.Activate(0.5);
                        results.Add((type.Name.Replace("`1", ""), closedType));
                    }
                    catch (NotSupportedException) { }
                }
                catch { }
            }
        }

        return results.OrderBy(r => r.Item1).ToList();
    }

    private static bool InheritsFromOpenGeneric(Type type, Type openGenericBase)
    {
        var current = type.BaseType;
        while (current is not null)
        {
            if (current.IsGenericType && current.GetGenericTypeDefinition() == openGenericBase)
                return true;
            current = current.BaseType;
        }
        return false;
    }

    /// <summary>
    /// All loss strategies for gradient checking, derived from the GradientCheckLossStrategy enum.
    /// Adding a new enum value automatically tests ALL layers with the new strategy.
    /// </summary>
    public static IEnumerable<object[]> LossStrategyValues =>
        ((GradientCheckLossStrategy[])Enum.GetValues(typeof(GradientCheckLossStrategy))).Select(s => new object[] { s });

    /// <summary>
    /// All scalar-compatible activation functions, auto-discovered via reflection.
    /// When new ActivationFunctionBase&lt;T&gt; implementations are added to the codebase,
    /// they automatically appear here and get tested with layers that support activation variants.
    /// </summary>
    public static IEnumerable<object[]> DiscoveredActivationNames =>
        _activationCache.Value.Select(a => new object[] { a.Name });


    // =========================================================================
    // Loss computation helpers
    // =========================================================================

    /// <summary>
    /// Computes a scalar loss value from the output tensor using the specified strategy.
    /// </summary>
    private static double ComputeStrategyLoss(Tensor<double> output, GradientCheckLossStrategy strategy)
    {
        switch (strategy)
        {
            case GradientCheckLossStrategy.MSE:
            {
                double loss = 0;
                for (int i = 0; i < output.Length; i++)
                    loss += output[i] * output[i];
                return loss / 2.0;
            }
            case GradientCheckLossStrategy.RandomProjection:
            {
                var rng = RandomHelper.CreateSeededRandom(12345);
                double loss = 0;
                for (int i = 0; i < output.Length; i++)
                    loss += (rng.NextDouble() * 2.0 - 1.0) * output[i];
                return loss;
            }
            case GradientCheckLossStrategy.Huber:
            {
                double loss = 0;
                for (int i = 0; i < output.Length; i++)
                {
                    double absVal = Math.Abs(output[i]);
                    loss += absVal < 1.0 ? 0.5 * output[i] * output[i] : absVal - 0.5;
                }
                return loss;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown loss strategy");
        }
    }

    /// <summary>
    /// Computes the gradient dL/dOutput for the specified loss strategy.
    /// </summary>
    private static Tensor<double> ComputeStrategyGradient(Tensor<double> output, GradientCheckLossStrategy strategy)
    {
        var grad = new Tensor<double>(output.Shape.ToArray());
        switch (strategy)
        {
            case GradientCheckLossStrategy.MSE:
                for (int i = 0; i < output.Length; i++)
                    grad[i] = output[i];
                break;
            case GradientCheckLossStrategy.RandomProjection:
            {
                var rng = RandomHelper.CreateSeededRandom(12345);
                for (int i = 0; i < output.Length; i++)
                    grad[i] = rng.NextDouble() * 2.0 - 1.0;
                break;
            }
            case GradientCheckLossStrategy.Huber:
                for (int i = 0; i < output.Length; i++)
                    grad[i] = Math.Abs(output[i]) < 1.0 ? output[i] : Math.Sign(output[i]);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown loss strategy");
        }
        return grad;
    }


    // =========================================================================
    // Tensor helpers
    // =========================================================================

    protected static Tensor<double> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0; // [-1, 1]
        return tensor;
    }

    protected static Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = value;
        return tensor;
    }


    // =========================================================================
    // INVARIANT 1: Forward produces finite, non-empty output
    // If the forward pass returns NaN/Inf/empty, the layer is numerically broken.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);

        Assert.True(output.Length > 0, "Layer output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN — numerical instability in Forward.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity — overflow in Forward.");
        }
    }

    // =========================================================================
    // INVARIANT 2: Forward is deterministic (same input -> same output)
    // Unless the layer has stochastic behavior (dropout), two calls with the
    // same input must produce bit-identical output.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        layer.SetTrainingMode(false); // Disable dropout/stochastic behavior
        var input = CreateRandomTensor(InputShape);

        var out1 = layer.Forward(input);
        layer.ResetState(); // Reset any recurrent state
        var out2 = layer.Forward(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
        {
            Assert.Equal(out1[i], out2[i]);
        }
    }

    // =========================================================================
    // INVARIANT 3: Different inputs produce different outputs
    // A layer that maps all inputs to the same output is broken (zero weights,
    // dead neurons, or input-ignoring forward pass).
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!ExpectsDifferentOutputForConstantInputs) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(false);

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        layer.ResetState();
        var output1 = layer.Forward(input1);
        layer.ResetState();
        var output2 = layer.Forward(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > Tolerance)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Layer produces identical output for inputs [0.1,...] and [0.9,...]. " +
            "Forward pass may ignore input values.");
    }

    // =========================================================================
    // INVARIANT 4: Output shape is consistent
    // GetOutputShape() must match the actual shape produced by Forward.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Forward_OutputShape_ShouldMatchGetOutputShape()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);
        var declaredShape = layer.GetOutputShape();

        // The output length should equal the product of declared output shape
        // (batch dimension may differ, so compare total feature count)
        int declaredFeatureCount = 1;
        foreach (var dim in declaredShape)
            declaredFeatureCount *= dim;

        // Allow for batch dimension: output.Length may be batch * declaredFeatureCount
        Assert.True(output.Length > 0, "Output should not be empty.");
        Assert.True(output.Length % declaredFeatureCount == 0 || declaredFeatureCount == 0,
            $"Output length {output.Length} is not a multiple of declared output shape " +
            $"[{string.Join(",", declaredShape)}] (product={declaredFeatureCount}).");
    }

    // =========================================================================
    // INVARIANT 5: (Removed — Backward deleted in tape-based autodiff migration)
    // Gradient correctness is now verified through GradientTape, not layer Backward.
    // =========================================================================

    // =========================================================================
    // INVARIANT 6: Parameter count is non-negative and GetParameters matches
    // ParameterCount must equal GetParameters().Length for trainable layers.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Parameters_CountShouldMatchVector()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();

        int count = layer.ParameterCount;
        var parameters = layer.GetParameters();

        Assert.True(count >= 0, "ParameterCount should be non-negative.");
        Assert.Equal(count, parameters.Length);

        if (ExpectsTrainableParameters)
        {
            Assert.True(count > 0,
                "Layer is expected to have trainable parameters but ParameterCount is 0.");
        }
    }

    // =========================================================================
    // INVARIANT 7: SetParameters -> GetParameters roundtrip
    // Setting parameters and getting them back should return the same values.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Parameters_SetGet_Roundtrip()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        if (layer.ParameterCount == 0) return; // Skip for non-trainable layers

        var original = layer.GetParameters();
        var modified = new Vector<double>(original.Length);
        for (int i = 0; i < original.Length; i++)
            modified[i] = original[i] + 0.001; // Small perturbation

        layer.SetParameters(modified);
        var retrieved = layer.GetParameters();

        Assert.Equal(modified.Length, retrieved.Length);
        for (int i = 0; i < modified.Length; i++)
        {
            Assert.Equal(modified[i], retrieved[i], 1e-15);
        }
    }

    // =========================================================================
    // INVARIANT 8: (Removed — Backward deleted in tape-based autodiff migration)
    // =========================================================================

    // =========================================================================
    // INVARIANT 9: Serialization roundtrip preserves behavior
    // Serialize -> Deserialize should produce a layer with identical Forward output.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Serialize_Deserialize_ShouldPreserveBehavior()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        layer.SetTrainingMode(false);
        var input = CreateRandomTensor(InputShape);

        var originalOutput = layer.Forward(input);

        // Serialize
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer.Serialize(writer);
        }

        // Deserialize into a fresh layer
        var layer2 = CreateLayer();
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer2.Deserialize(reader);
        }

        layer2.SetTrainingMode(false);
        layer2.ResetState();
        var deserializedOutput = layer2.Forward(input);

        Assert.Equal(originalOutput.Length, deserializedOutput.Length);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.True(Math.Abs(originalOutput[i] - deserializedOutput[i]) < 1e-12,
                $"Output[{i}] differs after serialization roundtrip: " +
                $"original={originalOutput[i]:G17}, deserialized={deserializedOutput[i]:G17}");
        }
    }

    // =========================================================================
    // INVARIANT 10: ResetState doesn't break the layer
    // After ResetState, Forward should still produce valid (finite) output.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task ResetState_ShouldNotBreakForward()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        // Forward once to populate state
        layer.Forward(input);

        // Reset
        layer.ResetState();

        // Forward again — should still work
        var output = layer.Forward(input);
        Assert.True(output.Length > 0, "Output should not be empty after ResetState.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN after ResetState + Forward.");
        }
    }

    // =========================================================================
    // INVARIANTS 11-14: (Removed — Backward deleted in tape-based autodiff migration)
    // Gradient correctness tests will be reimplemented using GradientTape<T>.
    // =========================================================================
}
