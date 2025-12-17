using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Stores intermediate layer activations collected during a forward pass.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> During neural network training, we often want to inspect or use
/// the outputs from internal layers (not just the final output). These internal outputs are called
/// "intermediate activations". This class stores them in a dictionary keyed by layer name.</para>
///
/// <para><b>Use Cases:</b>
/// - Feature-based distillation: Match intermediate layer outputs between teacher and student
/// - Neuron selectivity: Analyze how individual neurons respond across a batch
/// - Attention transfer: Transfer attention patterns from teacher to student
/// - Debugging: Inspect what each layer is learning</para>
///
/// <para><b>Example Usage:</b>
/// <code>
/// var activations = new IntermediateActivations&lt;double&gt;();
/// activations.Add("conv1", conv1Output);    // Store first conv layer output
/// activations.Add("conv2", conv2Output);    // Store second conv layer output
///
/// // Later, retrieve for analysis
/// var conv1Acts = activations.Get("conv1");
/// </code>
/// </para>
/// </remarks>
public class IntermediateActivations<T>
{
    private readonly Dictionary<string, Matrix<T>> _activations = new Dictionary<string, Matrix<T>>();

    /// <summary>
    /// Adds intermediate activations for a specific layer.
    /// </summary>
    /// <param name="layerName">The name or identifier of the layer (e.g., "conv1", "layer_3").</param>
    /// <param name="activations">The activation matrix for this layer. Rows = batch samples, Columns = neurons/features.</param>
    /// <remarks>
    /// <para>If the layer name already exists, it will be overwritten.</para>
    /// </remarks>
    public void Add(string layerName, Matrix<T> activations)
    {
        if (string.IsNullOrWhiteSpace(layerName))
            throw new System.ArgumentException("Layer name cannot be null or whitespace", nameof(layerName));
        if (activations == null)
            throw new System.ArgumentNullException(nameof(activations));

        _activations[layerName] = activations;
    }

    /// <summary>
    /// Retrieves intermediate activations for a specific layer.
    /// </summary>
    /// <param name="layerName">The name or identifier of the layer.</param>
    /// <returns>A defensive copy of the activation matrix, or null if the layer was not found.</returns>
    /// <remarks>
    /// Returns a cloned matrix to prevent external mutations from corrupting stored activations.
    /// </remarks>
    public Matrix<T>? Get(string layerName)
    {
        return _activations.TryGetValue(layerName, out var activations) ? activations.Clone() : null;
    }

    /// <summary>
    /// Gets all stored activations as a read-only dictionary.
    /// </summary>
    /// <remarks>
    /// <para>Key = layer name, Value = activation matrix.</para>
    /// <para><b>Note:</b> Returns defensive copies of matrices to prevent external mutation.</para>
    /// </remarks>
    public IReadOnlyDictionary<string, Matrix<T>> AllActivations =>
        _activations.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.Clone());

    /// <summary>
    /// Checks if activations exist for a specific layer.
    /// </summary>
    public bool Contains(string layerName)
    {
        return _activations.ContainsKey(layerName);
    }

    /// <summary>
    /// Gets the number of layers with stored activations.
    /// </summary>
    public int LayerCount => _activations.Count;
}
