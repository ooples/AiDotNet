using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
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
/// Base test class for ILayer&lt;T&gt; implementations.
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
public abstract class LayerTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a double literal into the fixture's numeric type.</summary>
    protected static T ToT(double value) => NumOps.FromDouble(value);

    /// <summary>Converts a fixture value to double for diagnostic calculations and assertions.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    /// <summary>
    /// Factory method — create a fresh instance of the layer under test.
    /// </summary>
    protected abstract ILayer<T> CreateLayer();

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
    protected virtual double Tolerance => typeof(T) == typeof(float) ? 1e-6 : 1e-12;

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
    /// Whether the layer's Forward output is expected to consist entirely of finite
    /// values. False for masking layers (ALiBi, causal attention masks) that emit
    /// ±Infinity at masked positions by design — the downstream softmax converts
    /// those to exact zero attention weight. The Forward_ShouldProduceFiniteOutput
    /// invariant skips the IsInfinity check when this is false. The TestScaffold
    /// generator emits an override of this from <c>[LayerProperty(ProducesNonFiniteOutput = true)]</c>.
    /// </summary>
    protected virtual bool ExpectsFiniteOutput => true;

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
    protected virtual ILayer<T> CreateLayerWithActivation(ActivationFunctionBase<T> activation)
        => CreateLayer();


    // =========================================================================
    // Static discovery infrastructure
    // Auto-discovers activation functions via reflection so that adding a new
    // ActivationFunctionBase<T> implementation automatically includes it in tests.
    // =========================================================================

    private static readonly Lazy<IReadOnlyList<(string Name, Type ClosedType)>> _activationCache =
        new(DiscoverScalarActivationTypes);

    /// <summary>
    /// Discovers all concrete ActivationFunctionBase&lt;T&gt; implementations that support
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
                    var closedType = type.MakeGenericType(typeof(T));
                    if (Activator.CreateInstance(closedType) is not ActivationFunctionBase<T> instance)
                        continue;

                    // Test scalar support by trying Activate — vector-only activations throw
                    try
                    {
                        instance.Activate(ToT(0.5));
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
    private static double ComputeStrategyLoss(Tensor<T> output, GradientCheckLossStrategy strategy)
    {
        switch (strategy)
        {
            case GradientCheckLossStrategy.MSE:
            {
                double loss = 0;
                for (int i = 0; i < output.Length; i++)
                {
                    double value = ToD(output[i]);
                    loss += value * value;
                }
                return loss / 2.0;
            }
            case GradientCheckLossStrategy.RandomProjection:
            {
                var rng = RandomHelper.CreateSeededRandom(12345);
                double loss = 0;
                for (int i = 0; i < output.Length; i++)
                    loss += (rng.NextDouble() * 2.0 - 1.0) * ToD(output[i]);
                return loss;
            }
            case GradientCheckLossStrategy.Huber:
            {
                double loss = 0;
                for (int i = 0; i < output.Length; i++)
                {
                    double value = ToD(output[i]);
                    double absVal = Math.Abs(value);
                    loss += absVal < 1.0 ? 0.5 * value * value : absVal - 0.5;
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
    private static Tensor<T> ComputeStrategyGradient(Tensor<T> output, GradientCheckLossStrategy strategy)
    {
        var grad = new Tensor<T>(output.Shape.ToArray());
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
                    grad[i] = ToT(rng.NextDouble() * 2.0 - 1.0);
                break;
            }
            case GradientCheckLossStrategy.Huber:
                for (int i = 0; i < output.Length; i++)
                {
                    double value = ToD(output[i]);
                    grad[i] = Math.Abs(value) < 1.0 ? output[i] : ToT(Math.Sign(value));
                }
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown loss strategy");
        }
        return grad;
    }


    // =========================================================================
    // Tensor helpers
    // =========================================================================

    protected static Tensor<T> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = ToT(rng.NextDouble() * 2.0 - 1.0); // [-1, 1]
        return tensor;
    }

    protected static Tensor<T> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = ToT(value);
        return tensor;
    }

    /// <summary>
    /// Creates deterministic test data from the layer's declared value-domain contract.
    /// Index consumers receive legal, varied integer IDs; all other layers retain the
    /// continuous random input used by the original conformance suite.
    /// </summary>
    protected static Tensor<T> CreateConformingInput(
        ILayer<T> layer, int[] shape, int seed = 42)
    {
        if (layer is not LayerBase<T> layerBase)
            return CreateRandomTensor(shape, seed);

        var contract = layerBase.BindInputContract(shape);
        contract.RequireReady();
        return InputContractTensorFactory.CreateValid<T>(
            shape,
            contract.PrimaryInput.ValueDomain,
            RandomHelper.CreateSeededRandom(seed));
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
        var input = CreateConformingInput(layer, InputShape);

        var output = layer.Forward(input);

        Assert.True(output.Length > 0, "Layer output should not be empty.");
        bool checkFinite = ExpectsFiniteOutput;
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(ToD(output[i])),
                $"Output[{i}] is NaN — numerical instability in Forward.");
            if (checkFinite)
            {
                Assert.False(double.IsInfinity(ToD(output[i])),
                    $"Output[{i}] is Infinity — overflow in Forward.");
            }
        }
    }

    // =========================================================================
    // INVARIANT 1b: A layer either HANDLES a nearby input shape or REJECTS it
    //               deliberately — it never crashes from the inside.
    // =========================================================================

    /// <summary>
    /// Feeds the layer input shapes one step away from its declared one, and requires every outcome to be
    /// either a real output or a deliberate rejection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// THE DISTINCTION IS THE WHOLE TEST. A layer is entitled to require a specific input shape — a patch
    /// embedding needs its extent divisible by the patch size, and saying so is correct behaviour. What a
    /// layer is not entitled to do is ASSUME a shape silently and then fail from inside a kernel with
    /// <c>IndexOutOfRangeException</c> or <c>NullReferenceException</c>. Those name no constraint, point
    /// at no caller mistake, and are indistinguishable from an engine defect; every one of them is a
    /// hard-coded assumption that was never written down.
    /// </para>
    /// <para>
    /// This is what makes shape support real rather than declared. A layer that works at 16x16 and
    /// crashes at 17x16 passes every other invariant in this file, because they all run at exactly one
    /// shape — which is precisely how such an assumption survives.
    /// </para>
    /// <para>
    /// The recovered shape relation is reported alongside any failure, because "what did this layer
    /// actually do to the shapes it accepted" is the first question after such a crash.
    /// </para>
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task Forward_NearbyShapes_AreHandledOrDeliberatelyRejected()
    {
        await Task.Yield();

        // NO OPT-OUT. There used to be a ShapeRobustnessApplicable hook, and an override to false
        // returned from this test having asserted NOTHING -- so the layers most likely to need this
        // invariant were the ones most likely to be excused from it, silently and permanently.
        // Every layer can participate, because the invariant already accepts an explicit rejection:
        // a layer whose input shape is genuinely fixed satisfies it by REFUSING nearby shapes with a
        // validation exception that names the constraint, which is the correct behaviour anyway.
        using var _arena = TensorArena.Create();

        var probes = AiDotNet.NeuralNetworks.ShapeRelationDiscovery.ProbeShapes(InputShape);
        var observations = new List<(int[] Input, int[] Output)>();
        var crashes = new List<string>();

        foreach (var shape in probes)
        {
            try
            {
                var candidateLayer = CreateLayer();
                var output = candidateLayer.Forward(CreateConformingInput(candidateLayer, shape));
                var outShape = output.Shape.ToArray();

                Assert.True(
                    outShape.Length > 0 && System.Array.TrueForAll(outShape, d => d > 0),
                    $"Input [{string.Join(",", shape)}] was accepted but produced the degenerate output "
                    + $"shape [{string.Join(",", outShape)}]. Accepting a shape and then emitting nothing "
                    + "is worse than rejecting it, because the caller gets no signal at all.");

                observations.Add((shape, outShape));
            }
            catch (Xunit.Sdk.XunitException)
            {
                throw;
            }
            catch (System.Exception ex) when (IsResourceExhaustion(ex))
            {
                // NOT a shape assumption. Running out of memory says the probe was too big for this
                // machine right now, which is a statement about the runner and not about the layer;
                // recording it as a hard-coded-shape defect would send the reader hunting for an
                // assumption that is not there. Rethrown so it surfaces as the resource failure it is.
                throw;
            }
            catch (System.Exception ex) when (IsDeliberateShapeRejection(ex))
            {
                // A stated constraint. Correct behaviour, and the caller is told what is wrong.
            }
            catch (System.Exception ex)
            {
                crashes.Add(
                    $"  input [{string.Join(",", shape)}] -> {ex.GetType().Name}: "
                    + $"{ex.Message.Split('\n')[0]}");
            }
        }

        if (crashes.Count > 0)
        {
            string relation = DescribeDiscoveredRelation(observations);
            Assert.Fail(
                $"{CreateLayer().GetType().Name} crashed from the INSIDE on input shapes one step away "
                + $"from its declared [{string.Join(",", InputShape)}]:\n"
                + string.Join("\n", crashes)
                + "\n\nThese are not shape validations — they name no constraint and point at no caller "
                + "mistake, so they read as engine defects. Either accept these shapes, or reject them "
                + "with an ArgumentException that says what this layer requires.\n"
                + $"Shape relation recovered from the shapes it DID accept: {relation}");
        }
    }

    /// <summary>Environmental failure - the machine, not the layer.</summary>
    private static bool IsResourceExhaustion(System.Exception ex)
        => ex is System.OutOfMemoryException or System.InsufficientExecutionStackException
            or System.OperationCanceledException;

    /// <summary>An exception that STATES a shape constraint, as opposed to one that leaks an assumption.</summary>
    /// <remarks>
    /// The AiDotNet.Exceptions shape family is listed FIRST because it is the best possible answer
    /// here, not a grudging allowance: a layer that throws TensorShapeMismatchException has not merely
    /// avoided crashing, it has named the exact constraint in a type built to carry it. A generic
    /// ArgumentException also passes, since stating the constraint in prose is still stating it.
    /// <para>
    /// THREE TYPES WERE REMOVED BECAUSE THEY SWALLOW THE DEFECT THIS INVARIANT HUNTS. The target is a
    /// layer that ASSUMES a shape silently and then fails from inside a kernel; accepting the generic
    /// failure modes as "deliberate rejection" meant exactly that failure counted as a pass:
    /// </para>
    /// <list type="bullet">
    /// <item>InvalidOperationException is accepted only when its message explicitly names a shape
    /// constraint. A generic kernel-state failure remains a crash.</item>
    /// <item>NotSupportedException and NotImplementedException say the layer does not do this at all.
    /// That is a gap in the layer, and marking it as a well-stated shape constraint hides it.</item>
    /// </list>
    /// <para>
    /// A layer that genuinely means to reject a shape has five precise types and ArgumentException
    /// available, all of which state the constraint.
    /// </para>
    /// </remarks>
    /// <summary>Whether an exception is the layer deliberately refusing a shape.</summary>
    /// <remarks>
    /// <para>
    /// The dedicated exception types are self-evidently shape rejections and are accepted outright.
    /// A bare <see cref="System.ArgumentException"/> is not: it is also what an internal argument
    /// failure throws, so accepting every one of them let a defect that names no shape constraint
    /// pass as correct behaviour -- which is the exact distinction this invariant exists to draw.
    /// </para>
    /// <para>
    /// So a generic ArgumentException has to SAY something about shape. The vocabulary below is the
    /// language layer validation actually uses; a message drawn from none of it is treated as a
    /// crash and reported, which is the safe direction: a real rejection worded outside this
    /// vocabulary shows up as a failure asking for a clearer message, whereas the old behaviour hid
    /// real defects.
    /// </para>
    /// </remarks>
    private static bool IsDeliberateShapeRejection(System.Exception ex)
    {
        if (ex is AiDotNet.Exceptions.TensorShapeMismatchException
            or AiDotNet.Exceptions.TensorDimensionException
            or AiDotNet.Exceptions.TensorRankException
            or AiDotNet.Exceptions.InvalidInputDimensionException
            or AiDotNet.Exceptions.VectorLengthMismatchException
            or System.RankException)
        {
            return true;
        }

        if (ex is not (System.ArgumentException or System.InvalidOperationException)) return false;

        return NamesAShapeConstraint(ex.Message);
    }

    /// <summary>The words a shape validation message uses when it states a constraint.</summary>
    /// <remarks>
    /// SHAPE-SPECIFIC ONLY. Generic words such as "expected", "size", "must be", "must have" and
    /// "mismatch" are intentionally absent. The phrases below name tensor axes/extents used by real
    /// layer validation, while bracketed B/C/H/W-style signatures are recognized separately.
    /// </remarks>
    private static readonly string[] ShapeConstraintVocabulary =
    {
        "shape", "dimension", "dimensions", "rank", "axis", "axes",
        "height", "width", "channel", "channels", "batch", "divisible", "spatial",
        "feature size", "feature dim", "input size", "inputdim", "modeldim", "encoderdim",
        "hiddensize", "head", "querylen", "keylen", "token", "octonion",
    };

    private static bool NamesAShapeConstraint(string? message)
    {
        if (string.IsNullOrWhiteSpace(message)) return false;

        foreach (var word in ShapeConstraintVocabulary)
        {
            if (message!.IndexOf(word, System.StringComparison.OrdinalIgnoreCase) >= 0) return true;
        }

        // Several image/video validators state their contract as an explicit tensor signature,
        // e.g. "Expected [B,F,3,8,8], got ...". The bracket is what makes this shape-specific;
        // plain "expected" remains insufficient and cannot hide an arbitrary argument failure.
        return message!.IndexOf("expected [", System.StringComparison.OrdinalIgnoreCase) >= 0
            || message.IndexOf("expects [", System.StringComparison.OrdinalIgnoreCase) >= 0;
    }

    /// <summary>Best-effort symbolic summary of what the layer did to the shapes it accepted.</summary>
    private static string DescribeDiscoveredRelation(List<(int[] Input, int[] Output)> observations)
    {
        if (observations.Count < 2) return "(too few accepted shapes to recover one)";

        int inRank = observations[0].Input.Length;
        int outRank = observations[0].Output.Length;
        if (observations.Any(o => o.Input.Length != inRank || o.Output.Length != outRank))
            return "(the accepted shapes do not share a rank, so no single relation describes them)";

        // Positional placeholders: this sweep runs on layers that mostly carry no axis-role annotation,
        // and inventing roles for them would be a claim the layer never made.
        var inAxes = Enumerable.Range(0, inRank).Select(i => (AiDotNet.Enums.TensorAxis)i).ToArray();
        var outAxes = Enumerable.Range(0, outRank).Select(i => (AiDotNet.Enums.TensorAxis)i).ToArray();

        try
        {
            var findings = AiDotNet.NeuralNetworks.ShapeRelationDiscovery.Fit(inAxes, outAxes, observations);
            return string.Join(
                ", ", findings.Select((f, i) => $"out[{i}] = {f.Relation?.ToString() ?? "?"}"));
        }
        catch
        {
            return "(relation recovery failed)";
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
        var input = CreateConformingInput(layer, InputShape);

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

        // A singleton index domain has only one legal value, so no conforming
        // second input exists with which to test value sensitivity.
        if (layer is LayerBase<T> layerBase)
        {
            var domain = layerBase.GetInputDomain(InputShape);
            if (domain.IsIndices && domain.MaxExclusive - domain.MinInclusive <= 1)
                return;
        }

        bool anyDifferent = false;
        // A single pair is probabilistic for randomly initialized networks with
        // saturating activations: two distinct inputs can both land in the same
        // dead ReLU region even though the implementation uses its input. Several
        // deterministic pairs keep the invariant strict for input-ignoring layers
        // without making a chance activation collision fail the whole family.
        for (int attempt = 0; attempt < 4 && !anyDifferent; attempt++)
        {
            int seed = 17 + attempt * 11;
            var input1 = CreateConformingInput(layer, InputShape, seed);
            // Adjacent offsets guarantee a different legal ID for every index
            // cardinality greater than one (unlike arbitrary seeds that can alias modulo N).
            var input2 = CreateConformingInput(layer, InputShape, seed + 1);

            layer.ResetState();
            var output1 = layer.Forward(input1);
            layer.ResetState();
            var output2 = layer.Forward(input2);

            int minLen = Math.Min(output1.Length, output2.Length);
            for (int i = 0; i < minLen; i++)
            {
                if (Math.Abs(ToD(output1[i]) - ToD(output2[i])) > Tolerance)
                {
                    anyDifferent = true;
                    break;
                }
            }
        }
        Assert.True(anyDifferent,
            "Layer produces identical output for several distinct inputs that conform to its declared value domain. " +
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
        var input = CreateConformingInput(layer, InputShape);

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

        // Drive lazy-shape resolution + weight allocation by running a
        // single Forward against InputShape. Without this, layers that
        // defer weight allocation to OnFirstForward (#1209) report
        // ParameterCount = 0 — which is correct lazy semantics, but
        // the invariant we're testing (count == GetParameters().Length
        // and >0 for trainable) requires the layer to be in its
        // "post first forward" state to be meaningful.
        using (var probeInput = CreateConformingInput(layer, InputShape, seed: 17))
        {
            try { layer.Forward(probeInput); }
            catch
            {
                // Single-input Forward failed — for dual-input layers
                // (DecoderLayer / TransformerDecoderLayer expecting
                // encoder output alongside decoder input) try the
                // params-based overload via reflection. The interface
                // only declares Forward(Tensor<T>); subclasses that
                // accept multiple tensors expose Forward(params Tensor<T>[]).
                try
                {
                    var paramsForward = layer.GetType().GetMethod(
                        "Forward",
                        new[] { typeof(Tensor<T>[]) });
                    paramsForward?.Invoke(layer, new object[] { new[] { probeInput, probeInput } });
                }
                catch
                {
                    // All probe shapes failed — the invariant still
                    // validates whatever state the ctor produced.
                    // Layers that can't be probed this way should
                    // override CreateLayer to return a pre-initialized
                    // instance.
                }
            }
        }

        // ParameterCount widened to long in #1244; cast for comparison
        // against Vector<T>.Length which is int-bounded.
        int count = (int)layer.ParameterCount;
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

        // Probe Forward to drive lazy-shape resolution + weight allocation.
        // Without this, lazy layers (#1209) report ParameterCount = 0 and the
        // roundtrip below would skip — which is correct lazy semantics, but
        // the invariant we're testing only has meaning post-resolution.
        using (var probeInput = CreateConformingInput(layer, InputShape, seed: 17))
        {
            try { layer.Forward(probeInput); } catch { }
        }

        if (layer.ParameterCount == 0) return; // Genuinely non-trainable layers.

        var original = layer.GetParameters();
        var modified = new Vector<T>(original.Length);
        for (int i = 0; i < original.Length; i++)
            modified[i] = NumOps.Add(original[i], ToT(0.001)); // Small perturbation

        layer.SetParameters(modified);
        var retrieved = layer.GetParameters();

        Assert.Equal(modified.Length, retrieved.Length);
        for (int i = 0; i < modified.Length; i++)
        {
            Assert.Equal(modified[i], retrieved[i]);
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
        var input = CreateConformingInput(layer, InputShape);

        // Keep the reference result outside the arena's reusable activation storage. Deep
        // composite forwards may legitimately recycle an earlier activation buffer during the
        // second layer's run; comparing a live arena view would then compare the restored output
        // with memory that the restored forward just overwrote, not with the original value.
        var originalOutput = layer.Forward(input).Clone();
        var originalParameters = layer.GetParameters();

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

        var restoredParameters = layer2.GetParameters();
        Assert.Equal(originalParameters.Length, restoredParameters.Length);
        for (int i = 0; i < originalParameters.Length; i++)
        {
            double originalParameter = ToD(originalParameters[i]);
            double restoredParameter = ToD(restoredParameters[i]);
            Assert.True(EqualityComparer<T>.Default.Equals(originalParameters[i], restoredParameters[i]),
                $"Parameter[{i}] differs after serialization roundtrip: " +
                $"original={originalParameter:G17}, deserialized={restoredParameter:G17}");
        }

        var originalTensors = AiDotNet.Training.TapeTrainingStep<T>
            .CollectParameters(new[] { layer }, structureVersion: -1);
        var restoredTensors = AiDotNet.Training.TapeTrainingStep<T>
            .CollectParameters(new[] { layer2 }, structureVersion: -1);
        Assert.Equal(originalTensors.Count, restoredTensors.Count);
        for (int tensorIndex = 0; tensorIndex < originalTensors.Count; tensorIndex++)
        {
            var originalTensor = originalTensors[tensorIndex];
            var restoredTensor = restoredTensors[tensorIndex];
            Assert.Equal(originalTensor.Shape.ToArray(), restoredTensor.Shape.ToArray());
            for (int i = 0; i < originalTensor.Length; i++)
            {
                double originalValue = ToD(originalTensor[i]);
                double restoredValue = ToD(restoredTensor[i]);
                Assert.True(EqualityComparer<T>.Default.Equals(originalTensor[i], restoredTensor[i]),
                    $"Trainable tensor {tensorIndex}[{i}] differs after serialization roundtrip: " +
                    $"original={originalValue:G17}, deserialized={restoredValue:G17}");
            }
        }

        var originalReplay = layer.Forward(input).Clone();
        for (int i = 0; i < originalOutput.Length; i++)
        {
            double originalValue = ToD(originalOutput[i]);
            double replayValue = ToD(originalReplay[i]);
            Assert.True(Math.Abs(originalValue - replayValue) < 1e-12,
                $"Serializing the layer changed its own output at [{i}]: " +
                $"before={originalValue:G17}, after={replayValue:G17}");
        }

        layer2.SetTrainingMode(false);
        layer2.ResetState();
        var deserializedOutput = layer2.Forward(input);

        Assert.Equal(originalOutput.Length, deserializedOutput.Length);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            // Direct equality check covers ±Infinity (where Math.Abs(inf - inf) = NaN
            // would make the tolerance check spuriously fail for layers like ALiBi that
            // legitimately emit -∞ at masked positions). For ordinary finite outputs the
            // direct comparison still requires bit-exact roundtrip because serialization
            // is lossless — fall back to the 1e-12 tolerance check only if they aren't
            // bit-equal so legacy near-equal serialization formats remain accepted.
            if (EqualityComparer<T>.Default.Equals(originalOutput[i], deserializedOutput[i]))
            {
                continue;
            }
            double originalValue = ToD(originalOutput[i]);
            double deserializedValue = ToD(deserializedOutput[i]);
            Assert.True(Math.Abs(originalValue - deserializedValue) < Tolerance,
                $"Output[{i}] differs after serialization roundtrip: " +
                $"original={originalValue:G17}, deserialized={deserializedValue:G17}");
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
        var input = CreateConformingInput(layer, InputShape);

        // Forward once to populate state
        layer.Forward(input);

        // Reset
        layer.ResetState();

        // Forward again — should still work
        var output = layer.Forward(input);
        Assert.True(output.Length > 0, "Output should not be empty after ResetState.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(ToD(output[i])),
                $"Output[{i}] is NaN after ResetState + Forward.");
        }
    }

    // =========================================================================
    // INVARIANT 11: Tape-based gradient flow — for layers with trainable
    // parameters, a real Engine-op-composed loss against the layer's forward
    // output must produce a non-zero gradient on at least one trainable
    // parameter. This is the modern equivalent of the pre-tape "Backward
    // produces non-zero parameter gradients" invariant. Catches tape-blocking
    // forward composition bugs (e.g. FlashAttention<T>.Forward filling its
    // output via scalar indexing, TensorMultiplyScalar in Forward paths,
    // or composite layers that fail to RegisterSubLayer their inner trainable
    // sub-layers).
    // =========================================================================

    [Fact(Timeout = 60000)]
    public async Task TapeGradient_ShouldReachAtLeastOneTrainableParameter()
    {
        await Task.Yield();
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        // Numerical derivatives require a deterministic function. Eval mode disables stochastic
        // masks/running-stat updates while preserving the differentiable layer transform.
        layer.SetTrainingMode(false);
        var input = CreateConformingInput(layer, InputShape);

        using var tape = new GradientTape<T>();
        var output = layer.Forward(input);

        // Use the production recursive collector, not only this layer's own tensor list. Composite
        // layers often own no tensors directly—their parameters live entirely in registered child
        // layers—so a local-only lookup made their generated gradient tests pass vacuously.
        // Collect AFTER Forward because lazy layers may replace zero-length placeholders.
        var trainableParams = AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(
            new[] { layer }, structureVersion: -1);
        if (trainableParams.Count == 0) return;

        // Tape-tracked random-projection loss: L = Σᵢ (output[i] · r[i]).
        // Engine.TensorMultiply + Engine.TensorSum are both standard tape-
        // tracked ops, so dL/doutput = r everywhere with no zero entries
        // (the RNG produces a [-1, 1] dense vector). Any trainable parameter
        // that's actually wired into the forward graph must therefore see
        // a non-zero gradient back-propagated to it.
        var projection = CreateRandomTensor(output.Shape.ToArray(), seed: 12345);
        var elementwise = AiDotNetEngine.Current.TensorMultiply(output, projection);
        // ReduceSum over all axes returns a tape-tracked scalar-rank-0 tensor;
        // Engine.TensorSum unwraps to a raw double which the tape can't consume.
        var allAxes = new int[elementwise.Shape.Length];
        for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
        var lossTensor = AiDotNetEngine.Current.ReduceSum(elementwise, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(lossTensor, trainableParams);

        bool foundNonZeroGrad = false;
        foreach (var kvp in grads)
        {
            var grad = kvp.Value;
            if (grad is null) continue;
            for (int i = 0; i < grad.Length; i++)
            {
                if (Math.Abs(ToD(grad[i])) > Tolerance)
                {
                    foundNonZeroGrad = true;
                    break;
                }
            }
            if (foundNonZeroGrad) break;
        }

        Assert.True(foundNonZeroGrad,
            "After Forward + tape-based ComputeGradients on a random-projection " +
            "loss, every trainable parameter received a zero gradient. The layer's " +
            "Forward composition is using Engine ops that don't propagate gradients " +
            "on the autodiff tape, OR a composite layer failed to register its " +
            "inner trainable sub-layers via RegisterSubLayer. Common causes: " +
            "Engine.TensorMultiplyScalar in Forward (not tape-tracked), " +
            "FlashAttention<T>.Forward (allocates output then fills via scalar " +
            "indexing — invisible to the tape), or `new Tensor<T>(...)` followed " +
            "by manual data fills inside a Forward override.");
    }

    // =========================================================================
    // INVARIANT 12: Numerical gradient correctness (finite differences).
    // For a sampled subset of trainable parameters (full sweep would be O(N×forward)
    // for N trainable scalars), verify the analytical gradient from the
    // autodiff tape matches the central-difference numerical gradient:
    //     dL/dw ≈ (L(w+ε) - L(w-ε)) / (2ε)
    // This is the gold standard for "did the backward implementation
    // correctly compute the derivative?" Layers that pass this AND the
    // existing forward-invariants 1-10 can be trusted to train.
    // =========================================================================

    [Fact(Timeout = 120000)]
    public async Task TapeGradient_ShouldMatchNumericalGradient()
    {
        await Task.Yield();
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        using var _arena = TensorArena.Create();
        var layer = CreateLayer();
        // Numerical derivatives require a deterministic eval-mode function. This also matches the
        // generated model gradcheck, so fused/eval operator paths cannot escape layer-level coverage.
        layer.SetTrainingMode(false);
        var input = CreateConformingInput(layer, InputShape);

        // --- Analytical gradient via tape ---
        using var tape = new GradientTape<T>();
        var output = layer.Forward(input);
        // Match the production training gateway's recursive ownership walk. A composite whose
        // tensors all live in children must be checked, not treated as parameter-free.
        var trainableParams = AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(
            new[] { layer }, structureVersion: -1);
        if (trainableParams.Count == 0) return;
        // Fix the projection BEFORE both gradient computations so the
        // analytical and numerical paths see the same loss surface.
        var projection = CreateRandomTensor(output.Shape.ToArray(), seed: 12345);
        var elementwise = AiDotNetEngine.Current.TensorMultiply(output, projection);
        // ReduceSum over all axes returns a tape-tracked scalar-rank-0 tensor;
        // Engine.TensorSum unwraps to a raw double which the tape can't consume.
        var allAxes = new int[elementwise.Shape.Length];
        for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
        var lossTensor = AiDotNetEngine.Current.ReduceSum(elementwise, allAxes, keepDims: false);
        // The generated/runtime input contract is the source of truth for differentiability.
        // Do not infer it from marker interfaces or per-fixture overrides: BERT/FastText/Layout
        // embedding front ends all consume integer IDs without implementing ITokenEmbedding,
        // while every ordinary continuous layer must retain a complete VJP to its input.
        var boundInput = ((LayerBase<T>)layer).BindInputContract(input.Shape.ToArray());
        boundInput.RequireReady();
        bool checkInputGradient =
            boundInput.PrimaryInput.ValueDomain.Kind == LayerInputDomainKind.Continuous;
        var gradientSources = new List<Tensor<T>>(trainableParams);
        if (checkInputGradient) gradientSources.Add(input);
        var analyticalGrads = tape.ComputeGradients(lossTensor, gradientSources);

        // --- Numerical gradient via central differences ---
        // Sample a small number of (param, index) pairs to keep the test
        // wall-time reasonable. A layer with broken backward will fail this
        // check on most sampled coordinates, so the sample doesn't need to
        // be exhaustive — it just needs to hit *some* trainable scalar.
        double eps = typeof(T) == typeof(float) ? 1e-3 : 1e-5;
        double numericalTolerance = typeof(T) == typeof(float) ? 5e-2 : 1e-3;
        const int MaxSampledPerParam = 6;

        int paramsChecked = 0;
        int paramsAgreed = 0;
        var deltas = new System.Text.StringBuilder();

        var rng = RandomHelper.CreateSeededRandom(7777);

        // SPARSE parameters (SparseLinearLayer's SparseTensor<T> weights) mirror
        // torch.autograd.gradcheck's "masked" semantics: it walks a sparse tensor's STORED (nnz)
        // entries via indices()/values() and perturbs only those, never densifying. Two reasons
        // this matters here rather than being a style choice:
        //   1. Flat indexing a SparseTensor throws outright — "GetFlat is not supported on sparse
        //      tensors. Use SparseTensor-specific APIs or call ToDense() first." — so the check
        //      cannot even run against one.
        //   2. Densifying would be WORSE than the crash: it would perturb STRUCTURAL ZEROS, which
        //      are not trainable parameters and therefore have no analytical gradient, producing
        //      false mismatches and an inflated parameter count. SparseLinearLayer's own
        //      ParameterCount is NonZeroCount + OutputFeatures, confirming the stored values are
        //      the trainable set.
        // Dense parameters keep the exact previous behaviour (flat index over Length).
        // Access the sparse payload through DataVector.AsSpan(), NOT through the Values property:
        // `public T[] Values => DataVector.ToArray()` allocates a FRESH COPY on every access, so the
        // earlier `sp.Values[i] = v` wrote into a throwaway array and never perturbed the parameter.
        // Both finite-difference evaluations therefore saw identical weights, making the numerical
        // gradient exactly 0 for every sparse entry while the analytical gradient was non-zero —
        // which is what produced "disagrees ... on 5/12 sampled trainable scalars" (the entries that
        // "agreed" were simply the ones whose analytical gradient was also ~0). Same copy-versus-view
        // trap as Tensor<T>.ToVector(); AsSpan() is the documented zero-copy path.
        static int TrainableScalarCount(Tensor<T> p) =>
            p is SparseTensor<T> sp ? sp.NonZeroCount : p.Length;
        static double ReadScalar(Tensor<T> p, int i) =>
            ToD(p is SparseTensor<T> sp ? sp.DataVector[i] : p[i]);
        static void WriteScalar(Tensor<T> p, int i, double v)
        {
            if (p is SparseTensor<T> sp) sp.DataVector[i] = ToT(v);
            else p[i] = ToT(v);
        }
        // The analytical gradient of a SPARSE parameter is not necessarily sparse. When it comes back
        // DENSE, index i (a position in the sparse nnz payload) addresses a completely different
        // matrix entry in a dense flat buffer, so comparing them positionally checks unrelated
        // numbers. Map through the COO coordinates instead. A sparse gradient shares the parameter's
        // payload layout, so it is read directly.
        // THE PUBLIC SPARSE API, NOT THE INTERNAL PAYLOAD. This read the backing DataVector of
        // the very type under test, so a change to that payload's layout would silently change
        // what the gradient check compares -- the helper would keep returning a number and the
        // number would mean something else. SparseTensor<T> exposes Values / RowIndices /
        // ColumnIndices, which is what the SparseTensor suites themselves assert against.
        //
        // COO construction is rank-2, so the column stride is Shape[1]; a non-rank-2 sparse
        // tensor has no COO reading and is refused rather than indexed on a guess.
        static double ReadAnalyticalScalar(Tensor<T> grad, Tensor<T> param, int i)
        {
            if (grad is SparseTensor<T> gsp)
            {
                return i >= 0 && i < gsp.Values.Length ? ToD(gsp.Values[i]) : 0.0;
            }

            if (param is SparseTensor<T> psp)
            {
                Assert.True(psp.Shape.Length == 2,
                    $"SparseTensor COO indices are rank-2; got rank {psp.Shape.Length}, so " +
                    "RowIndices/ColumnIndices cannot be mapped to a flat gradient index.");
                if (i < 0 || i >= psp.RowIndices.Length) return 0.0;

                int cols = psp.Shape[1];
                int flat = (psp.RowIndices[i] * cols) + psp.ColumnIndices[i];
                return flat >= 0 && flat < grad.Length ? ToD(grad[flat]) : 0.0;
            }

            return ToD(grad[i]);
        }

        foreach (var param in trainableParams)
        {
            if (param is null || TrainableScalarCount(param) == 0) continue;
            if (!analyticalGrads.TryGetValue(param, out var analyticalGrad) || analyticalGrad is null)
                continue;

            int trainableCount = TrainableScalarCount(param);
            int sampleCount = Math.Min(MaxSampledPerParam, trainableCount);
            for (int s = 0; s < sampleCount; s++)
            {
                int idx = rng.Next(0, trainableCount);

                double original = ReadScalar(param, idx);
                double plusValue = ToD(ToT(original + eps));
                double minusValue = ToD(ToT(original - eps));
                if (plusValue == minusValue) continue;

                WriteScalar(param, idx, plusValue);
                var lossPlus = ComputeProjectionLossScalar(layer.Forward(input), projection);
                WriteScalar(param, idx, minusValue);
                var lossMinus = ComputeProjectionLossScalar(layer.Forward(input), projection);
                WriteScalar(param, idx, original);

                // Divide by the ACTUAL representable perturbation. For float, original ± eps
                // can round asymmetrically (or together); pretending the denominator is exactly
                // 2*eps manufactures a gradient error in the test harness itself.
                double numerical = (lossPlus - lossMinus) / (plusValue - minusValue);
                double analytical = ReadAnalyticalScalar(analyticalGrad, param, idx);
                double absDiff = Math.Abs(numerical - analytical);
                double scale = Math.Max(Math.Max(Math.Abs(numerical), Math.Abs(analytical)), 1.0);

                paramsChecked++;
                if (absDiff / scale < numericalTolerance)
                {
                    paramsAgreed++;
                }
                else if (deltas.Length < 1000)
                {
                    deltas.Append($"  idx={idx} numerical={numerical:G6} analytical={analytical:G6} reldiff={absDiff / scale:G3}\n");
                }
            }
        }

        // A layer's VJP has two equally important consumers: its own parameters and every layer
        // before it. Parameter-only gradchecks allowed a broken input derivative to remain invisible
        // until a full model happened to place the operator downstream. Validate a deterministic
        // sample of the continuous input here so every generated layer fixture covers both halves of
        // the reverse-mode contract automatically.
        if (checkInputGradient)
        {
            Assert.True(analyticalGrads.TryGetValue(input, out var inputGradient) && inputGradient is not null,
                "The layer exposes trainable tape gradients but its continuous input is disconnected " +
                "from the reverse-mode graph. Mark a genuinely discrete/control input explicitly; do " +
                "not let a missing input VJP silently pass parameter-only conformance.");

            int inputSamples = Math.Min(12, input.Length);
            int inputAgreed = 0;
            var inputDeltas = new System.Text.StringBuilder();
            for (int sample = 0; sample < inputSamples; sample++)
            {
                int index = inputSamples == input.Length
                    ? sample
                    : (sample * Math.Max(1, input.Length / inputSamples)) % input.Length;
                double original = ToD(input[index]);
                double plusValue = ToD(ToT(original + eps));
                double minusValue = ToD(ToT(original - eps));
                if (plusValue == minusValue) continue;

                input[index] = ToT(plusValue);
                double lossPlus = ComputeProjectionLossScalar(layer.Forward(input), projection);
                input[index] = ToT(minusValue);
                double lossMinus = ComputeProjectionLossScalar(layer.Forward(input), projection);
                input[index] = ToT(original);

                double numerical = (lossPlus - lossMinus) / (plusValue - minusValue);
                double analytical = ToD(inputGradient[index]);
                double difference = Math.Abs(numerical - analytical);
                double scale = Math.Max(Math.Max(Math.Abs(numerical), Math.Abs(analytical)), 1.0);
                if (difference / scale < numericalTolerance)
                {
                    inputAgreed++;
                }
                else if (inputDeltas.Length < 1000)
                {
                    inputDeltas.Append(
                        $"  input[{index}] numerical={numerical:G6} analytical={analytical:G6} " +
                        $"reldiff={difference / scale:G3}\n");
                }
            }

            Assert.True(inputAgreed * 3 >= inputSamples * 2,
                $"Tape-based input VJP disagrees with finite differences on " +
                $"{inputSamples - inputAgreed}/{inputSamples} sampled input scalars. First mismatches:\n" +
                inputDeltas +
                "The layer may compute its own parameter gradients correctly while propagating an " +
                "incorrect gradient to every preceding layer.");
        }

        if (paramsChecked == 0) return; // no comparable parameter scalars
        // At least 2/3 of sampled params must agree. A more lenient threshold
        // tolerates layers that intentionally produce different-shaped
        // gradients (e.g. STE-style surrogates) — those should set
        // ExpectsNonZeroGradients=false anyway, so reaching this assertion
        // already means the layer claims paper-faithful backward.
        Assert.True(paramsAgreed * 3 >= paramsChecked * 2,
            $"Tape-based analytical gradient disagrees with finite-difference " +
            $"numerical gradient on {paramsChecked - paramsAgreed}/{paramsChecked} " +
            $"sampled trainable scalars. First mismatches:\n{deltas}" +
            "Likely the layer's Forward composition records the wrong derivative " +
            "for some Engine op, OR a non-tape-tracked op is silently used as " +
            "an identity for the gradient.");

        // Cover every registered trainable tensor in one normalized direction. Coordinate sampling
        // localizes defects cheaply; this complementary JVP-style check prevents an entire parameter
        // slot from escaping merely because none of its scalar indices happened to be sampled.
        var direction = new List<(Tensor<T> Parameter, Tensor<T> Gradient, int Index, double Sign)>();
        for (int parameterIndex = 0; parameterIndex < trainableParams.Count; parameterIndex++)
        {
            var parameter = trainableParams[parameterIndex];
            if (parameter is null || TrainableScalarCount(parameter) == 0 ||
                !analyticalGrads.TryGetValue(parameter, out var gradient) || gradient is null)
                continue;

            int count = TrainableScalarCount(parameter);
            int index = (parameterIndex * 7919 + 17) % count;
            direction.Add((parameter, gradient, index, (parameterIndex & 1) == 0 ? 1.0 : -1.0));
        }
        if (direction.Count == 0) return;

        double directionScale = 1.0 / Math.Sqrt(direction.Count);
        double directionalStep = eps / directionScale;
        double analyticalDirection = 0.0;
        foreach (var coordinate in direction)
        {
            analyticalDirection += ReadAnalyticalScalar(
                coordinate.Gradient,
                coordinate.Parameter,
                coordinate.Index) * coordinate.Sign * directionScale;
        }

        double EvaluateDirection(double step, double sign)
        {
            try
            {
                foreach (var coordinate in direction)
                {
                    double original = ReadScalar(coordinate.Parameter, coordinate.Index);
                    WriteScalar(
                        coordinate.Parameter,
                        coordinate.Index,
                        original + (sign * step * directionScale * coordinate.Sign));
                }
                return ComputeProjectionLossScalar(layer.Forward(input), projection);
            }
            finally
            {
                // Each evaluation starts from the exact same parameter state.
                foreach (var coordinate in direction)
                {
                    double perturbed = ReadScalar(coordinate.Parameter, coordinate.Index);
                    WriteScalar(
                        coordinate.Parameter,
                        coordinate.Index,
                        perturbed - (sign * step * directionScale * coordinate.Sign));
                }
            }
        }

        double plus = EvaluateDirection(directionalStep, +1.0);
        double minus = EvaluateDirection(directionalStep, -1.0);
        // CLARKE SUBDIFFERENTIAL, not a central difference against a single number.
        //
        // The tape does not always compute "the" derivative, because for some layers one does not
        // exist. Jaderberg et al., "Spatial Transformer Networks" (NeurIPS 2015) Sec 3.3, defining
        // the bilinear sampler this suite exercises through every warping layer:
        //
        //     "Due to discontinuities in the sampling functions, sub-gradients must be used."
        //
        // and its prescribed derivative picks a value AT the kink by convention:
        //
        //     dV/dx = SUM U * max(0, 1-|y-n|) * { 0 if |m-x| >= 1 ; 1 if m >= x ; -1 if m < x }
        //
        // So the analytical value is a SUB-gradient by design. A central difference straddling a
        // breakpoint converges to a chord across it -- a different mathematical object -- and
        // comparing the two is a category error rather than a test. Measured on
        // SVTRThinPlateSplineLayer's _controlWeights against a fixed analytical -5.5954, the central
        // difference does not converge as h shrinks; it wanders and then inverts:
        //
        //     h=1e-2 -> -3.704   h=1e-3 -> -2.958   h=1e-4 -> -1.004
        //     h=1e-5 -> -2.882   h=1e-6 -> +3.598   h=1e-7 -> -55.693
        //
        // A real scale error in a gradient holds a constant ratio under that sweep. This flips sign,
        // which is the signature of the reference being wrong rather than the tape.
        //
        // For a piecewise-smooth f the correct object is the Clarke subdifferential: at a kink the
        // derivative is the INTERVAL spanned by the one-sided derivatives, and any value inside it
        // is a valid sub-gradient. So bracket rather than compare. On a smooth f the two one-sided
        // derivatives coincide, the interval collapses to a point, and this is exactly as strict as
        // a central-difference equality check -- no coverage is traded away for the smooth layers,
        // and the non-smooth ones gain a correct assertion where they previously had a wrong one.
        //
        // Richardson extrapolation is deliberately not used. It cancels the O(h^2) Taylor term,
        // which a piecewise-LINEAR function does not have, so across a breakpoint it amplifies the
        // disagreement between step sizes instead of cancelling it. PyTorch's gradcheck likewise
        // uses a plain central difference and no extrapolation.
        double baseline = ComputeProjectionLossScalar(layer.Forward(input), projection);
        double forwardOneSided = (plus - baseline) / directionalStep;
        double backwardOneSided = (baseline - minus) / directionalStep;
        double lower = Math.Min(forwardOneSided, backwardOneSided);
        double upper = Math.Max(forwardOneSided, backwardOneSided);

        // The bracket must still be tolerant: each one-sided difference carries O(h) truncation
        // error plus floating-point noise, so a smooth layer's analytical value can sit marginally
        // outside a bracket that has effectively zero width.
        double bracketScale = Math.Max(
            Math.Max(Math.Abs(lower), Math.Abs(upper)),
            Math.Max(Math.Abs(analyticalDirection), 1.0));
        double slack = bracketScale * numericalTolerance * 2.0;

        bool insideBracket =
            analyticalDirection >= lower - slack && analyticalDirection <= upper + slack;

        Assert.True(insideBracket,
            $"Directional gradient across {direction.Count} trainable parameter tensors is not a " +
            $"valid sub-gradient: analytical={analyticalDirection:G8} lies outside the one-sided " +
            $"bracket [{lower:G8}, {upper:G8}] (slack {slack:G4}). For a smooth layer the two " +
            "one-sided derivatives coincide, so this is an ordinary gradient mismatch; for a layer " +
            "that warps through a bilinear sampler the bracket is the Clarke subdifferential and " +
            "the analytical value must lie inside it to be a legitimate sub-gradient.");
    }

    /// <summary>
    /// Scalar projection-loss for the finite-difference path. Mirrors what
    /// the tape-side TapeGradient_ShouldMatchNumericalGradient computes via
    /// Engine.TensorMultiply + Engine.TensorSum, but on detached output
    /// (no tape recording — we just want the scalar value for L(w±ε)).
    /// </summary>
    private static double ComputeProjectionLossScalar(Tensor<T> output, Tensor<T> projection)
    {
        double sum = 0;
        int len = Math.Min(output.Length, projection.Length);
        for (int i = 0; i < len; i++) sum += ToD(output[i]) * ToD(projection[i]);
        return sum;
    }
}

/// <summary>Default-precision alias for existing hand-written fixtures.</summary>
public abstract class LayerTestBase : LayerTestBase<double> { }
