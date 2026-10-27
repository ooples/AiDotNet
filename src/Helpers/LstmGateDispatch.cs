using System.Reflection;
using System.Runtime.CompilerServices;

namespace AiDotNet.Helpers;

/// <summary>Uses the tensor engine's fused gate operation when installed, preserving compatibility
/// with older tensor packages that expose only the dense sequence operation.</summary>
internal static class LstmGateDispatch<T>
{
    private delegate Tensor<T> Kernel(Tensor<T> gates, Tensor<T> previousCell, out Tensor<T> cell);
    private sealed class Binding
    {
        internal readonly Kernel? Apply;
        internal Binding(IEngine engine)
        {
            var method = engine.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "LstmCellFromGates" && m.IsGenericMethodDefinition
                    && m.GetGenericArguments().Length == 1 && m.GetParameters().Length == 3);
            if (method is not null)
                Apply = (Kernel)Delegate.CreateDelegate(typeof(Kernel), engine, method.MakeGenericMethod(typeof(T)));
        }
    }
    private static readonly ConditionalWeakTable<IEngine, Binding> Bindings = new();

    internal static Tensor<T> Forward(IEngine engine, Tensor<T> gates, Tensor<T> previousCell, out Tensor<T> cell)
    {
        var kernel = Bindings.GetValue(engine, e => new Binding(e)).Apply;
        if (kernel is not null) return kernel(gates, previousCell, out cell);
        int width = previousCell.Shape[previousCell.Rank - 1];
        Tensor<T> Gate(int i) => engine.TensorNarrow(gates, gates.Rank - 1, i * width, width);
        cell = engine.TensorAdd(engine.TensorMultiply(engine.Sigmoid(Gate(1)), previousCell),
            engine.TensorMultiply(engine.Sigmoid(Gate(0)), engine.Tanh(Gate(2))));
        return engine.TensorMultiply(engine.Sigmoid(Gate(3)), engine.Tanh(cell));
    }
}
