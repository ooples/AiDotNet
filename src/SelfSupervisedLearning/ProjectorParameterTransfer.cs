using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>Copies or averages live projector tensors without flattening weights onto the host.</summary>
internal static class ProjectorParameterTransfer<T>
{
    internal static void Copy(IProjectorHead<T> source, IProjectorHead<T> destination, bool leadingProjectorOnly = false)
        => Update(source, destination, 0, leadingProjectorOnly);

    internal static void Update(IProjectorHead<T> source, IProjectorHead<T> destination,
        double momentum, bool leadingProjectorOnly = false)
    {
        if (double.IsNaN(momentum) || momentum < 0 || momentum > 1) throw new ArgumentOutOfRangeException(nameof(momentum));
        var from = source.GetTrainableTensors();
        var to = destination.GetTrainableTensors();
        if (from.Count < to.Count || (!leadingProjectorOnly && from.Count != to.Count))
            throw new ArgumentException("Projector parameter layouts differ.", nameof(source));
        for (int i = 0; i < to.Count; i++)
            if (!from[i].Shape.ToArray().SequenceEqual(to[i].Shape.ToArray()))
                throw new ArgumentException($"Projector tensor {i} has an incompatible shape.", nameof(source));
        using var noGrad = new NoGradScope<T>();
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        // Compute every source before writing: even aliased or reordered live tensors are safe.
        var updated = momentum == 0
            ? TensorCheckpoint<T>.Capture(from.Take(to.Count).ToArray())
            : to.Select((tensor, i) => engine.TensorAdd(
                engine.TensorMultiplyScalar(tensor, ops.FromDouble(momentum)),
                engine.TensorMultiplyScalar(from[i], ops.FromDouble(1 - momentum)))).ToArray();
        TensorCheckpoint<T>.Restore(to, updated);
    }
}
