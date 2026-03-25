using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

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
    /// Whether the layer's Backward produces meaningful gradients.
    /// Some layers (ReservoirLayer, InputLayer) pass gradients through but
    /// don't compute weight gradients. Override to false for those.
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
        Enum.GetValues<GradientCheckLossStrategy>().Select(s => new object[] { s });

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
    // Shared gradient check logic
    // =========================================================================

    /// <summary>
    /// Runs numerical gradient check (finite differences) against analytical gradients
    /// for a given layer, input, and loss strategy. Returns failure count and details.
    /// </summary>
    private static (int FailCount, int CheckCount, string DebugInfo) RunGradientCheck(
        ILayer<double> layer, Tensor<double> input, GradientCheckLossStrategy lossStrategy, double epsilon = 1e-5)
    {
        layer.SetTrainingMode(true);
        layer.ClearGradients();
        var output = layer.Forward(input);

        var outputGrad = ComputeStrategyGradient(output, lossStrategy);
        layer.Backward(outputGrad);
        var analyticalGradients = layer.GetParameterGradients();

        if (analyticalGradients.Length == 0)
            return (0, 0, string.Empty);

        var parameters = layer.GetParameters();
        int checkCount = Math.Min(10, parameters.Length);
        int failCount = 0;
        var debugInfo = new System.Text.StringBuilder();

        for (int p = 0; p < checkCount; p++)
        {
            // L(w + epsilon)
            var paramsPlus = parameters.Clone();
            paramsPlus[p] += epsilon;
            layer.SetParameters(paramsPlus);
            layer.ResetState();
            var outputPlus = layer.Forward(input);
            double lossPlus = ComputeStrategyLoss(outputPlus, lossStrategy);

            // L(w - epsilon)
            var paramsMinus = parameters.Clone();
            paramsMinus[p] -= epsilon;
            layer.SetParameters(paramsMinus);
            layer.ResetState();
            var outputMinus = layer.Forward(input);
            double lossMinus = ComputeStrategyLoss(outputMinus, lossStrategy);

            double numericalGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
            double analyticalGrad = analyticalGradients[p];

            // Relative error check (handle near-zero gradients)
            double absMax = Math.Max(Math.Abs(numericalGrad), Math.Abs(analyticalGrad));
            if (absMax < 1e-7) continue; // Both near zero, skip

            double relError = Math.Abs(numericalGrad - analyticalGrad) / (absMax + 1e-8);
            if (relError > 0.01) // 1% relative error threshold
            {
                failCount++;
                debugInfo.Append(
                    $"p[{p}]: analytical={analyticalGrad:G6} numerical={numericalGrad:G6} relErr={relError:G4} | ");
            }
        }

        // Restore original parameters
        layer.SetParameters(parameters);
        return (failCount, checkCount, debugInfo.ToString());
    }


    // =========================================================================
    // INVARIANT 1: Forward produces finite, non-empty output
    // If the forward pass returns NaN/Inf/empty, the layer is numerically broken.
    // =========================================================================

    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
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

    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
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

    [Fact]
    public void Forward_DifferentInputs_ShouldProduceDifferentOutputs()
    {
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

    [Fact]
    public void Forward_OutputShape_ShouldMatchGetOutputShape()
    {
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
    // INVARIANT 5: Backward produces finite gradient
    // After Forward+Backward, the returned input gradient must be finite.
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceFiniteGradient()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);

        // Create gradient matching output shape
        var outputGrad = CreateRandomTensor(output.Shape.ToArray(), seed: 99);

        var inputGrad = layer.Backward(outputGrad);

        Assert.True(inputGrad.Length > 0, "Input gradient should not be empty.");
        for (int i = 0; i < inputGrad.Length; i++)
        {
            Assert.False(double.IsNaN(inputGrad[i]),
                $"InputGradient[{i}] is NaN — broken backward pass.");
            Assert.False(double.IsInfinity(inputGrad[i]),
                $"InputGradient[{i}] is Infinity — gradient explosion in backward pass.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Parameter count is non-negative and GetParameters matches
    // ParameterCount must equal GetParameters().Length for trainable layers.
    // =========================================================================

    [Fact]
    public void Parameters_CountShouldMatchVector()
    {
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

    [Fact]
    public void Parameters_SetGet_Roundtrip()
    {
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
    // INVARIANT 8: Backward produces non-zero weight gradients
    // For trainable layers, after Forward+Backward, GetParameterGradients
    // should have at least some non-zero values.
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceNonZeroWeightGradients()
    {
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        layer.ClearGradients();

        var input = CreateRandomTensor(InputShape);
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape.ToArray(), seed: 99);

        layer.Backward(outputGrad);

        var gradients = layer.GetParameterGradients();
        Assert.True(gradients.Length > 0,
            "Trainable layer should have parameter gradients after Backward.");

        bool anyNonZero = false;
        for (int i = 0; i < gradients.Length; i++)
        {
            if (Math.Abs(gradients[i]) > 1e-15)
            {
                anyNonZero = true;
                break;
            }
        }
        Assert.True(anyNonZero,
            "All parameter gradients are zero after Backward. " +
            "Gradient computation is likely broken.");
    }

    // =========================================================================
    // INVARIANT 9: Serialization roundtrip preserves behavior
    // Serialize -> Deserialize should produce a layer with identical Forward output.
    // =========================================================================

    [Fact]
    public void Serialize_Deserialize_ShouldPreserveBehavior()
    {
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

    [Fact]
    public void ResetState_ShouldNotBreakForward()
    {
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
    // INVARIANT 11: ClearGradients zeroes all gradients
    // After ClearGradients, GetParameterGradients should return all zeros.
    // =========================================================================

    [Fact]
    public void ClearGradients_ShouldZeroAllGradients()
    {
        if (!ExpectsTrainableParameters) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(true);

        // Accumulate some gradients
        var input = CreateRandomTensor(InputShape);
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape.ToArray(), seed: 99);
        layer.Backward(outputGrad);

        // Clear
        layer.ClearGradients();

        var gradients = layer.GetParameterGradients();
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(0.0, gradients[i], 1e-15);
        }
    }

    // =========================================================================
    // INVARIANT 12: Numerical gradient check (finite differences)
    // For trainable layers, verify that analytical gradients match numerical
    // approximation: dL/dw = (L(w+e) - L(w-e)) / 2e
    // This is the gold standard for gradient correctness.
    // Uses the loss strategy specified by DefaultLossStrategy.
    // =========================================================================

    [Fact]
    public void Backward_NumericalGradientCheck()
    {
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        var (failCount, checkCount, debugInfo) = RunGradientCheck(layer, input, DefaultLossStrategy);
        if (checkCount == 0) return;

        Assert.True(failCount <= checkCount / 3,
            $"Numerical gradient check failed for {failCount}/{checkCount} parameters. " +
            $"Analytical gradients don't match finite differences (loss={DefaultLossStrategy}). " +
            $"Details: {debugInfo}");
    }

    // =========================================================================
    // INVARIANT 13: Loss variant gradient check
    // Tests the backward pass with every loss strategy to expose bugs hidden by
    // specific gradient alignments. Each strategy produces a different dL/dOutput:
    // - MSE: gradient proportional to output (bugs cancel when backward * output)
    // - RandomProjection: random gradient direction (no alignment with output)
    // - L1: constant magnitude gradient (tests backward with sign-only signal)
    // Adding a new entry to LossStrategyNames automatically tests ALL layers.
    // =========================================================================

    [Theory]
    [Trait("Category", "GradientVariant")]
    [MemberData(nameof(LossStrategyValues), MemberType = typeof(LayerTestBase))]
    public void Backward_GradientCheck_LossVariant(GradientCheckLossStrategy lossStrategy)
    {
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        var (failCount, checkCount, debugInfo) = RunGradientCheck(layer, input, lossStrategy);
        if (checkCount == 0) return;

        Assert.True(failCount <= checkCount / 3,
            $"Gradient check ({lossStrategy}) failed for {failCount}/{checkCount} parameters. " +
            $"Details: {debugInfo}");
    }

    // =========================================================================
    // INVARIANT 14: Activation variant gradient check
    // For layers that accept activation function parameters, tests gradient
    // correctness with every auto-discovered scalar activation function.
    // Adding a new ActivationFunctionBase<T> implementation automatically tests it.
    // Layers opt in by overriding SupportsActivationVariants => true and
    // implementing CreateLayerWithActivation().
    // =========================================================================

    [Theory]
    [Trait("Category", "GradientVariant")]
    [MemberData(nameof(DiscoveredActivationNames), MemberType = typeof(LayerTestBase))]
    public void Backward_GradientCheck_ActivationVariant(string activationName)
    {
        if (!SupportsActivationVariants) return;
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var match = _activationCache.Value.FirstOrDefault(a => a.Name == activationName);
        if (match.ClosedType is null) return;

        if (Activator.CreateInstance(match.ClosedType) is not ActivationFunctionBase<double> activation)
            return;

        var layer = CreateLayerWithActivation(activation);
        var input = CreateRandomTensor(InputShape);

        var (failCount, checkCount, debugInfo) = RunGradientCheck(layer, input, GradientCheckLossStrategy.MSE);
        if (checkCount == 0) return;

        Assert.True(failCount <= checkCount / 3,
            $"Gradient check with {activationName} failed for {failCount}/{checkCount} parameters. " +
            $"Details: {debugInfo}");
    }
}
