using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Training;

namespace AiDotNet.ComputerVision;

/// <summary>
/// Tape-based training for the computer-vision models that are built on
/// <c>ModelBase&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c> rather than <c>NeuralNetworkBase</c> -
/// the object detectors, text detectors and OCR recognizers.
/// </summary>
/// <remarks>
/// <para>
/// Those models hold their layers as discrete private fields (often behind the <c>Conv2D</c> /
/// <c>Dense</c> adapters in <c>BackboneLayerShims</c>) instead of an enumerable layer collection,
/// so there is no <c>Layers</c> property to hand to <see cref="TapeTrainingStep{T}"/>. This helper
/// recovers that list by walking the object graph for <see cref="ITrainableLayer{T}"/> instances,
/// which is what makes the existing tape trainer usable here instead of a second hand-written
/// training loop.
/// </para>
/// <para>
/// The walk is cached per model instance: the field graph is fixed once a model is constructed,
/// and repeating a reflection walk on every training step would dominate the step cost.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the model is expressed in.</typeparam>
internal static class TensorModelTrainer<T>
{
    /// <summary>
    /// Per-model-instance cache of the collected layers. A weak table so caching a model here
    /// never keeps it alive.
    /// </summary>
    private static readonly ConditionalWeakTable<object, IReadOnlyList<ITrainableLayer<T>>> LayerCache = new();

    /// <summary>
    /// Models whose lazy layers have already resolved their shapes, so the warm-up forward is
    /// paid once per model rather than on every training step.
    /// </summary>
    private static readonly ConditionalWeakTable<object, object> Warmed = new();

    /// <summary>
    /// Depth bound for the field walk. The deepest real chain is
    /// model -> backbone -> stage -> block -> shim -> layer, so this is generous.
    /// </summary>
    private const int MaxWalkDepth = 12;

    /// <summary>
    /// Runs one tape-based training step: forward under a gradient tape, mean-squared-error loss,
    /// then a stochastic-gradient update of every trainable tensor the walk found.
    /// </summary>
    /// <param name="model">The model being trained; the root of the field walk.</param>
    /// <param name="input">The training input.</param>
    /// <param name="target">The desired output, shaped like the model prediction.</param>
    /// <param name="learningRate">Step size for the parameter update.</param>
    /// <param name="forward">
    /// The model's differentiable forward pass. It must be built from engine operations so the
    /// tape records it - a forward that drops to scalar loops severs the chain and the parameters
    /// upstream of the break receive no gradient.
    /// </param>
    /// <returns>The loss value for this step.</returns>
    public static T Step(
        object model,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward)
    {
        // Resolve lazy layer shapes BEFORE collecting. The convolution layers behind the Conv2D
        // shim infer their input depth on first Forward and report no trainable parameters until
        // they have: collecting first would return an empty set and the step would silently do
        // nothing. No tape is active here, so this records nothing -- and it is paid once per
        // model, not once per step.
        if (!Warmed.TryGetValue(model, out _))
        {
            forward(input);
            Warmed.Add(model, model);
        }

        var layers = GetTrainableLayers(model);
        if (layers.Count == 0)
        {
            return MathHelper.GetNumericOperations<T>().Zero;
        }

        return TapeTrainingStep<T>.Step(
            layers,
            input,
            target,
            learningRate,
            forward,
            MeanSquaredError);
    }

    /// <summary>
    /// Mean squared error built from engine operations so the gradient tape can differentiate it.
    /// </summary>
    private static Tensor<T> MeanSquaredError(Tensor<T> predicted, Tensor<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        var difference = engine.TensorSubtract(predicted, target);
        var squared = engine.TensorMultiply(difference, difference);
        return engine.TensorMultiplyScalar(
            engine.ReduceSum(squared, null),
            numOps.FromDouble(1.0 / Math.Max(1, squared.Length)));
    }

    /// <summary>
    /// Returns the trainable layers reachable from the model, collecting them on first use.
    /// </summary>
    public static IReadOnlyList<ITrainableLayer<T>> GetTrainableLayers(object model)
        => LayerCache.GetValue(model, static root => Collect(root));

    private static IReadOnlyList<ITrainableLayer<T>> Collect(object root)
    {
        var found = new List<ITrainableLayer<T>>();
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        Walk(root, found, seen, 0);
        return found;
    }

    private static void Walk(object? node, List<ITrainableLayer<T>> found, HashSet<object> seen, int depth)
    {
        if (node is null || depth > MaxWalkDepth || !seen.Add(node))
        {
            return;
        }

        if (node is ITrainableLayer<T> trainable)
        {
            found.Add(trainable);

            // Do not descend into a layer. Composite layers own their sub-layers' parameters
            // through their own GetTrainableParameters, so walking in would add the same tensors
            // twice, and TapeTrainingStep would then apply the update to them twice.
            return;
        }

        // Collections of layers (a detection head is usually List<Conv2D<T>>).
        if (node is IEnumerable sequence and not string)
        {
            foreach (var element in sequence)
            {
                if (element is not null && !IsLeaf(element.GetType()))
                {
                    Walk(element, found, seen, depth + 1);
                }
            }

            return;
        }

        foreach (var field in EnumerateFields(node.GetType()))
        {
            if (IsLeaf(field.FieldType))
            {
                continue;
            }

            object? value;
            try
            {
                value = field.GetValue(node);
            }
            catch (TargetInvocationException)
            {
                continue; // A property-backed field that throws before initialization.
            }

            Walk(value, found, seen, depth + 1);
        }
    }

    private static IEnumerable<FieldInfo> EnumerateFields(Type type)
    {
        for (Type? current = type; current is not null && IsWalkable(current); current = current.BaseType)
        {
            foreach (var field in current.GetFields(
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                yield return field;
            }
        }
    }

    /// <summary>
    /// Types whose fields are worth walking: our own, excluding the tensor library, whose types
    /// are storage rather than model structure and whose internals are large.
    /// </summary>
    private static bool IsWalkable(Type type)
    {
        string? ns = type.Namespace;
        return ns is not null
            && ns.StartsWith("AiDotNet", StringComparison.Ordinal)
            && !ns.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal);
    }

    /// <summary>
    /// Types the walk must not descend into: primitives, strings, and the tensor and vector
    /// storage types that would otherwise be enumerated element by element.
    /// </summary>
    private static bool IsLeaf(Type type)
        => type.IsPrimitive
        || type.IsEnum
        || type == typeof(string)
        || type == typeof(decimal)
        || type == typeof(DateTime)
        || type == typeof(TimeSpan)
        || typeof(Delegate).IsAssignableFrom(type)
        || (type.Namespace is not null
            && type.Namespace.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal)
            && !typeof(ITrainableLayer<T>).IsAssignableFrom(type));

    private sealed class ReferenceEqualityComparer : IEqualityComparer<object>
    {
        public static readonly ReferenceEqualityComparer Instance = new();

        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);

        public int GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
