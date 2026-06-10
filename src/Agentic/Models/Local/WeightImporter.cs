using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Imports named weight tensors from an <see cref="INamedTensorSource"/> (safetensors or GGUF) into an
/// AiDotNet network's flat parameter vector. The caller supplies the tensor names in the network's parameter
/// order; the importer concatenates their values and calls <c>SetParameters</c>.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet networks expose a single flat parameter vector (<c>GetParameters</c>/<c>SetParameters</c>) rather
/// than named per-layer tensors, so the architecture-specific part is the <em>ordering</em> of source tensor
/// names that matches that flat layout — supplied by the caller (a per-architecture name list). This class is
/// the format-agnostic import mechanism; a built-in name map for a concrete architecture is a follow-up that
/// belongs with that model.
/// </para>
/// <para><b>For Beginners:</b> Once you know the order a model lays out its weights, hand that list of tensor
/// names plus a loaded file to this importer and it pours the weights into the model.
/// </para>
/// </remarks>
public static class WeightImporter
{
    /// <summary>
    /// Imports the named tensors (in the given order) into the model's parameters.
    /// </summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="model">The network to load weights into.</param>
    /// <param name="orderedTensorNames">The source tensor names in the model's flat parameter order.</param>
    /// <param name="source">The loaded weight source.</param>
    /// <exception cref="ArgumentNullException">Thrown when any argument is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the total imported parameter count does not match the model.</exception>
    public static void ImportInto<T>(NeuralNetworkBase<T> model, IReadOnlyList<string> orderedTensorNames, INamedTensorSource source)
    {
        Guard.NotNull(model);
        Guard.NotNull(orderedTensorNames);
        Guard.NotNull(source);

        var values = new List<T>();
        foreach (var name in orderedTensorNames)
        {
            foreach (var value in source.ReadAsDouble(name))
            {
                values.Add((T)Convert.ChangeType(value, typeof(T)));
            }
        }

        if (values.Count != model.ParameterCount)
        {
            throw new InvalidOperationException(
                $"Imported {values.Count} parameters but the model expects {model.ParameterCount}. " +
                "Check the tensor name ordering matches the architecture's parameter layout.");
        }

        model.SetParameters(new Vector<T>(values.ToArray()));
    }
}
