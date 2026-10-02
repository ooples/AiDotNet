using System.Collections;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// PyTorch's <c>torch.optim.swa_utils.update_bn</c> for models that are not a <c>NeuralNetworkBase</c> (the
/// vision task families). After a short run the momentum average of BatchNorm's running statistics still
/// carries the statistics of earlier weights. An eval-mode measurement of the current weights therefore needs
/// statistics re-estimated for them. For example, CRAFT's 13 VGG-BN layers gave a region peak of 0.93 in
/// training mode and 0.19 in eval mode after 40 Adam steps, although the loss had converged.
/// </summary>
internal static class BatchNormRecalibration
{
    /// <summary>
    /// Runs <paramref name="forward"/> once with every BatchNorm layer reachable from <paramref name="model"/>
    /// replacing its running statistics with the batch's. Layer modes are left alone.
    /// </summary>
    public static void Recalibrate<T>(object model, Action forward)
    {
        var layers = new List<BatchNormalizationLayer<T>>();
        Collect(model, layers, new HashSet<object>(ReferenceEqualityComparer.Instance));
        foreach (var layer in layers) layer.OverwriteRunningStatistics = true;
        try { forward(); }
        finally { foreach (var layer in layers) layer.OverwriteRunningStatistics = false; }
    }

    // Walks the model's own object graph: fields of AiDotNet types, and the collections and tuples holding them.
    private static void Collect<T>(object? node, List<BatchNormalizationLayer<T>> found, HashSet<object> seen)
    {
        // Tensor-library values (tensors, vectors, matrices) hold numbers, never layers, and can hold millions.
        if (node is null || node is string || node.GetType().IsPrimitive
            || node.GetType().Namespace?.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal) == true
            || !seen.Add(node)) return;
        if (node is BatchNormalizationLayer<T> batchNorm) found.Add(batchNorm);
        if (node is IEnumerable sequence && node is not IDictionary)
        {
            foreach (var item in sequence) Collect(item, found, seen);
            return;
        }
        var type = node.GetType();
        bool tuple = type.FullName?.StartsWith("System.ValueTuple", StringComparison.Ordinal) == true
            || type.FullName?.StartsWith("System.Tuple", StringComparison.Ordinal) == true;
        if (!tuple && type.Namespace?.StartsWith("AiDotNet", StringComparison.Ordinal) != true) return;
        for (var current = type; current is not null && current != typeof(object); current = current.BaseType)
            foreach (var field in current.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                if (!field.FieldType.IsPrimitive && !field.FieldType.IsEnum)
                    Collect(field.GetValue(node), found, seen);
    }
}
