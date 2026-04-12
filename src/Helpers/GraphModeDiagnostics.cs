using System.Text;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Diagnostic helper for dumping GraphMode compiled graphs.
/// Uses InternalsVisibleTo access to Tensors internals.
/// </summary>
internal static class GraphModeDiagnostics
{
    internal static string DumpCompiledGraph(
        Tensor<float>[] parameters,
        Tensor<float> input,
        Tensor<float> target,
        Action traceAction)
    {
        var paramSet = new HashSet<object>(parameters.Select(p => (object)p));
        paramSet.Add(input);
        var sb = new StringBuilder();

        using var scope = GraphMode.Enable();
        traceAction();

        var nodes = scope.Nodes;
        sb.AppendLine($"Total nodes: {nodes.Count}");

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];
            if (node is LazyNode<float> typed)
            {
                sb.Append($"[{i}] {typed.OpName} out=[{string.Join(",", typed.OutputShape)}]");
                var inputs = typed.GetInputsArray();
                sb.Append($" inputs={inputs.Length}");
                for (int j = 0; j < inputs.Length; j++)
                {
                    var inp = inputs[j];
                    bool isParam = parameters.Any(p => ReferenceEquals(p, inp));
                    bool isInput = ReferenceEquals(inp, input);
                    bool isTarget = ReferenceEquals(inp, target);
                    string tag = isParam ? "*PARAM*" : isInput ? "*INPUT*" : isTarget ? "*TARGET*" : "";
                    sb.Append($" in{j}=[{string.Join(",", inp._shape)}]{tag}");
                }
                sb.Append(typed.BackwardFn != null ? " BWD=yes" : " BWD=NO");
                sb.AppendLine();
            }
            else
            {
                sb.AppendLine($"[{i}] {node.OpType} out=[{string.Join(",", node.OutputShape)}] (non-float)");
            }
        }

        scope.MarkCompiled(); // Prevent auto-realize on dispose
        return sb.ToString();
    }
}
