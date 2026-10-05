using System.Reflection;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Constructor arguments for the model sweeps in this folder, which build every model of a family to check a property
/// of its structure or shape contract.
/// </summary>
/// <remarks>
/// Options are built size-bounded (ModelTestScale, the generated bound the clone sweep uses) instead of at their paper
/// defaults. A model's layer types and its shape laws do not depend on its width or depth, but its memory does: built at
/// paper scale, ModelInputMeetsFirstLayerTests reached 50.9 GB and ForecastOutputRankDiagnosticTests 8.2 GB in one test
/// host, and the Unassigned - 01 shard that runs them lost its 16 GB runner on every branch. Size knobs are scaled
/// together and the knobs that change the work done (hop, stride, patch) are never touched, so a sweep observes the same
/// structure. An options type the generator cannot bound keeps its default.
/// </remarks>
internal static class ModelSweepConstruction
{
    /// <summary>The arguments for an architecture-first constructor whose other parameters all have defaults.</summary>
    internal static object?[] Arguments(ParameterInfo[] parameters, object architecture)
    {
        var args = new object?[parameters.Length];
        args[0] = architecture;
        for (int i = 1; i < parameters.Length; i++)
        {
            var type = Nullable.GetUnderlyingType(parameters[i].ParameterType) ?? parameters[i].ParameterType;
            args[i] = type.IsClass && type != typeof(string)
                ? AiDotNet.Testing.ModelTestScale.CreateBoundedOptions(type) ?? parameters[i].DefaultValue
                : parameters[i].DefaultValue;
        }
        return args;
    }
}
