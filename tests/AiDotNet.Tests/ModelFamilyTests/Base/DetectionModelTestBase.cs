using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Shared base for the computer-vision detection and OCR families: object detection
/// (<see cref="ObjectDetectionTestBase{T}"/>), text detection
/// (<see cref="TextDetectionTestBase{T}"/>) and text recognition (<see cref="OCRTestBase{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// These models are <c>IFullModel&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c> built on
/// <c>ModelBase</c>, NOT <c>INeuralNetworkModel</c>: they hold their layers as discrete private
/// fields rather than an enumerable layer collection, and expose no architecture object. So they
/// cannot use <c>NeuralNetworkModelTestBase</c>, whose invariants are written against
/// <c>Layers</c>, <c>GetArchitecture()</c> and <c>GetNamedLayerActivations()</c>. This base
/// carries the part of the contract they DO share - the <c>IFullModel</c> surface - and each
/// domain base adds the invariants specific to its output format.
/// </para>
/// <para>
/// Two of the invariants here exist because the shared contract was silently broken:
/// <see cref="Train_ShouldChangeParameters"/> pins that training does something at all, and
/// <see cref="Clone_ShouldNotShareParameterStorage"/> pins that a clone owns its own weights.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the model is expressed in.</typeparam>
public abstract class DetectionModelTestBase<T>
    where T : struct
{
    /// <summary>Numeric operations for <typeparamref name="T"/>.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a double to <typeparamref name="T"/>.</summary>
    protected static T ToT(double value) => NumOps.FromDouble(value);

    /// <summary>Converts a <typeparamref name="T"/> to double.</summary>
    protected static double ToD(T value) => NumOps.ToDouble(value);

    /// <summary>
    /// Builds the model under test. The generated fixtures override this.
    /// </summary>
    protected abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateModel();

    /// <summary>
    /// Shape of the image tensor fed to the model, as NCHW. Detection and OCR backbones are
    /// convolutional and downsample by up to 32x, so the spatial dimensions must stay a multiple
    /// of 32 for the feature-pyramid levels to line up.
    /// </summary>
    protected virtual int[] InputShape => [1, 3, 64, 64];

    /// <summary>
    /// Number of training steps the training invariants take. Detection losses are slow per step,
    /// so this stays small; the invariants assert that something changed, not that the model
    /// converged.
    /// </summary>
    protected virtual int TrainingIterations => 2;

    /// <summary>
    /// Creates a deterministic pseudo-random image in [0, 1], the range the detector
    /// preprocessing expects.
    /// </summary>
    protected Tensor<T> CreateRandomImage(Random rng)
    {
        var tensor = new Tensor<T>(InputShape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = ToT(rng.NextDouble());
        }

        return tensor;
    }

    /// <summary>
    /// Creates a target tensor shaped like the model output, for the training invariants.
    /// </summary>
    protected Tensor<T> CreateTargetLike(Tensor<T> output, Random rng)
    {
        var target = new Tensor<T>(output._shape);
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = ToT(rng.NextDouble());
        }

        return target;
    }


    /// <summary>
    /// Runs one forward pass so lazy layers resolve their shapes.
    /// </summary>
    /// <remarks>
    /// The convolutions behind the <c>Conv2D</c> adapter infer their input depth on first
    /// <c>Forward</c> and report NO trainable parameters until they have. Reading
    /// <c>GetParameters()</c> on a freshly constructed model therefore sees the backbone only, and
    /// the count grows the moment anything runs a forward - which would make a before/after
    /// parameter comparison compare two different lengths.
    /// </remarks>
    protected void WarmUp(IFullModel<T, Tensor<T>, Tensor<T>> model, Random rng)
        => model.Predict(CreateRandomImage(rng));

    private static Vector<T> ParametersOf(IFullModel<T, Tensor<T>, Tensor<T>> model)
        => ((IParameterizable<T, Tensor<T>, Tensor<T>>)model).GetParameters();

    [Fact(Timeout = 120000)]
    public async Task ForwardPass_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var output = model.Predict(CreateRandomImage(rng));

        Assert.True(output.Length > 0, "Model produced an empty output tensor.");
        for (int i = 0; i < output.Length; i++)
        {
            double value = ToD(output[i]);
            Assert.False(double.IsNaN(value), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(value), $"Output[{i}] is Infinity.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var image = CreateRandomImage(rng);

        var first = model.Predict(image);
        var second = model.Predict(image);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(ToD(first[i]), ToD(second[i]), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var first = model.Predict(CreateRandomImage(rng));
        var second = model.Predict(CreateRandomImage(rng));

        // A model whose output ignores its input is not reading the image at all - the failure
        // mode a constant-returning stub would show. A two-stage detector's output length depends on
        // how many proposals survive, so a different LENGTH already proves input dependence.
        if (first.Length != second.Length)
        {
            return;
        }

        bool anyDifference = false;
        for (int i = 0; i < first.Length && !anyDifference; i++)
        {
            if (Math.Abs(ToD(first[i]) - ToD(second[i])) > 1e-12)
            {
                anyDifference = true;
            }
        }

        Assert.True(anyDifference, "Two different images produced byte-identical output.");
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        WarmUp(model, rng);

        Assert.True(ParametersOf(model).Length > 0, "Model reports no trainable parameters.");
    }

    [Fact(Timeout = 120000)]
    public async Task Metadata_ShouldExist()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldProduceIdenticalOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var image = CreateRandomImage(rng);

        var clone = model.Clone();

        var original = model.Predict(image);
        var copied = clone.Predict(image);

        Assert.Equal(original.Length, copied.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(ToD(original[i]), ToD(copied[i]), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldNotShareParameterStorage()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        WarmUp(model, rng);

        var before = ParametersOf(model);
        Assert.True(before.Length > 0, "Model reports no trainable parameters.");

        var clone = model.Clone();

        // Perturb the CLONE. A clone that shares weight storage with its original - the classic
        // MemberwiseClone shallow copy - will drag the original along with it, and every
        // downstream user who cloned a model to fine-tune it would silently corrupt the source.
        var mutated = new Vector<T>(before.Length);
        for (int i = 0; i < before.Length; i++)
        {
            mutated[i] = NumOps.Add(before[i], ToT(1.0));
        }

        ((IParameterizable<T, Tensor<T>, Tensor<T>>)clone).SetParameters(mutated);

        var after = ParametersOf(model);
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(ToD(before[i]), ToD(after[i]), 10);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Train_ShouldChangeParameters()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var image = CreateRandomImage(rng);
        WarmUp(model, rng);

        var before = ParametersOf(model);
        Assert.True(before.Length > 0, "Model reports no trainable parameters.");

        for (int step = 0; step < TrainingIterations; step++)
        {
            // Re-derive the target each step: a two-stage detector's output length follows its
            // proposals, which move once its weights do.
            model.Train(image, CreateTargetLike(model.Predict(image), rng));
        }

        var after = ParametersOf(model);
        Assert.Equal(before.Length, after.Length);

        bool anyChange = false;
        for (int i = 0; i < before.Length && !anyChange; i++)
        {
            if (Math.Abs(ToD(before[i]) - ToD(after[i])) > 1e-12)
            {
                anyChange = true;
            }
        }

        Assert.True(
            anyChange,
            "Train() left every parameter untouched. The model cannot learn: either the training "
            + "step is a no-op or no gradient reaches the parameters.");
    }

    [Fact(Timeout = 300000)]
    public async Task Train_ShouldProduceFinitePredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var image = CreateRandomImage(rng);
        for (int step = 0; step < TrainingIterations; step++)
        {
            model.Train(image, CreateTargetLike(model.Predict(image), rng));
        }

        var output = model.Predict(image);
        for (int i = 0; i < output.Length; i++)
        {
            double value = ToD(output[i]);
            Assert.False(double.IsNaN(value), $"Output[{i}] is NaN after training.");
            Assert.False(double.IsInfinity(value), $"Output[{i}] is Infinity after training.");
        }
    }

    // =====================================================
    // REGISTRATION AUDIT
    // The parameter registry is the single source of truth for GetParameters, Serialize, DeepCopy
    // AND training: the trainer updates exactly the registry's live trainable chunks. So a weight the
    // registry cannot see is silently never saved, cloned or trained. These two invariants close
    // that loop from both sides.
    // =====================================================

    /// <summary>
    /// Stable ids of registered trainable tensors that legitimately receive no gradient from the
    /// model's forward pass (for example a training-only auxiliary head). Empty by default: an
    /// unexplained untouched weight is a defect.
    /// </summary>
    protected virtual IReadOnlyCollection<string> ParametersUnusedByForward => System.Array.Empty<string>();

    [Fact(Timeout = 180000)]
    public async Task EveryTrainableLayerTensor_ShouldBeRegisteredLive()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        WarmUp(model, rng);

        var registered = new HashSet<Tensor<T>>(ReferenceComparer.Instance);
        foreach (var chunk in ((ModelBase<T, Tensor<T>, Tensor<T>>)model).GetParameterStateChunks())
        {
            if (chunk.Role == ParameterSlotRole.Trainable && chunk.IsWritableInPlace)
            {
                registered.Add(chunk.Tensor);
            }
        }

        var missing = new List<string>();
        foreach (var (path, layer) in ReachableTrainableLayers(model))
        {
            var tensors = layer.GetTrainableParameters();
            for (int i = 0; i < tensors.Count; i++)
            {
                if (tensors[i] is not null && tensors[i].Length > 0 && !registered.Contains(tensors[i]))
                {
                    missing.Add($"{path} ({layer.GetType().Name}) tensor #{i} [{string.Join(",", tensors[i].Shape.ToArray())}]");
                }
            }
        }

        Assert.True(
            missing.Count == 0,
            $"{missing.Count} trainable layer tensor(s) are reachable from the model but are not live, "
            + "trainable chunks of its parameter registry, so they are never saved, cloned or trained:\n  "
            + string.Join("\n  ", missing.Take(25)));
    }

    [Fact(Timeout = 300000)]
    public async Task Train_ShouldUpdateEveryRegisteredTrainableTensor()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var image = CreateRandomImage(rng);
        var target = CreateTargetLike(model.Predict(image), rng);

        var chunks = ((ModelBase<T, Tensor<T>, Tensor<T>>)model).GetParameterStateChunks()
            .Where(c => c.Role == ParameterSlotRole.Trainable && c.IsWritableInPlace && c.Tensor.Length > 0)
            .ToList();
        Assert.NotEmpty(chunks);

        var before = chunks.Select(c => { var snap = new double[c.Tensor.Length]; for (int i = 0; i < snap.Length; i++) snap[i] = ToD(c.Tensor[i]); return snap; }).ToList();
        model.Train(image, target);

        var untouched = new List<string>();
        for (int k = 0; k < chunks.Count; k++)
        {
            if (ParametersUnusedByForward.Contains(chunks[k].StableId))
            {
                continue;
            }

            var after = chunks[k].Tensor;
            bool moved = false;
            for (int i = 0; i < after.Length && !moved; i++)
            {
                moved = ToD(after[i]) != before[k][i];
            }

            if (!moved)
            {
                untouched.Add($"{chunks[k].StableId} [{string.Join(",", after.Shape.ToArray())}]");
            }
        }

        Assert.True(
            untouched.Count == 0,
            $"{untouched.Count} of {chunks.Count} registered trainable tensors did not move after a "
            + "training step. Either no gradient reaches them (the forward pass severs the autodiff tape "
            + "upstream of them) or they are dead weights the forward never reads:\n  "
            + string.Join("\n  ", untouched.Take(25)));
    }

    private static IEnumerable<(string Path, ITrainableLayer<T> Layer)> ReachableTrainableLayers(object root)
    {
        var seen = new HashSet<object>(ReferenceComparer.Instance);
        var found = new List<(string, ITrainableLayer<T>)>();
        Walk(root, root.GetType().Name, found, seen, 0);
        return found;
    }

    private static void Walk(object? node, string path, List<(string, ITrainableLayer<T>)> found, HashSet<object> seen, int depth)
    {
        if (node is null || depth > 14 || !seen.Add(node))
        {
            return;
        }

        if (node is ITrainableLayer<T> layer)
        {
            // Composite layers own their sub-layers' tensors through their own
            // GetTrainableParameters, so the walk stops at the first layer it meets.
            found.Add((path, layer));
            return;
        }

        if (node is IEnumerable sequence and not string)
        {
            int index = 0;
            foreach (var element in sequence)
            {
                if (element is not null && !IsLeaf(element.GetType()))
                {
                    Walk(element, $"{path}[{index}]", found, seen, depth + 1);
                }

                index++;
            }

            return;
        }

        for (var type = node.GetType(); type is not null && IsAiDotNetType(type); type = type.BaseType)
        {
            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (IsLeaf(field.FieldType))
                {
                    continue;
                }

                Walk(field.GetValue(node), $"{path}.{field.Name}", found, seen, depth + 1);
            }
        }
    }

    private static bool IsAiDotNetType(Type type)
        => type.Namespace is not null
        && type.Namespace.StartsWith("AiDotNet", StringComparison.Ordinal)
        && !type.Namespace.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal);

    private static bool IsLeaf(Type type)
        => type.IsPrimitive || type.IsEnum || type == typeof(string) || typeof(Delegate).IsAssignableFrom(type)
        || (type.Namespace is not null && type.Namespace.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal)
            && !typeof(ITrainableLayer<T>).IsAssignableFrom(type));

    private sealed class ReferenceComparer : IEqualityComparer<object>, IEqualityComparer<Tensor<T>>
    {
        public static readonly ReferenceComparer Instance = new();

        bool IEqualityComparer<object>.Equals(object? x, object? y) => ReferenceEquals(x, y);

        int IEqualityComparer<object>.GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);

        public bool Equals(Tensor<T>? x, Tensor<T>? y) => ReferenceEquals(x, y);

        public int GetHashCode(Tensor<T> obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
