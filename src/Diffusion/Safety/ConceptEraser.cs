using AiDotNet.Tensors.Engines;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Safety;

/// <summary>
/// Concept erasure for removing unwanted concepts from diffusion model representations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Concept erasure removes specific concepts (e.g., artistic styles, identities, or
/// unsafe content) from a diffusion model's internal representations by projecting
/// embeddings onto the null space of the target concept direction. This prevents the
/// model from generating content related to the erased concept.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine the model's understanding of concepts as directions
/// in a high-dimensional space. Concept erasure finds the "direction" corresponding to
/// an unwanted concept and removes it, like erasing one axis from a coordinate system.
/// The model can still generate everything else, but it loses the ability to produce
/// the erased concept.
/// </para>
/// <para>
/// Reference: Gandikota et al., "Erasing Concepts from Diffusion Models", ICCV 2023
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class ConceptEraser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly List<Vector<T>> _conceptDirections;
    private readonly double _erasureStrength;
    private readonly bool _preserveGeneralCapability;

    /// <summary>
    /// Initializes a new concept eraser.
    /// </summary>
    /// <param name="erasureStrength">Strength of concept removal (0.0 = none, 1.0 = full removal). Default: 1.0.</param>
    /// <param name="preserveGeneralCapability">Whether to preserve model capability on non-erased concepts. Default: true.</param>
    public ConceptEraser(
        double erasureStrength = 1.0,
        bool preserveGeneralCapability = true)
    {
        _conceptDirections = [];
        _erasureStrength = erasureStrength;
        _preserveGeneralCapability = preserveGeneralCapability;
    }

    /// <summary>
    /// Registers a concept direction to be erased.
    /// </summary>
    /// <param name="conceptEmbedding">Embedding vector representing the concept to erase.</param>
    public void AddConceptDirection(Vector<T> conceptEmbedding)
    {
        // Normalize the concept direction
        var normalized = NormalizeVector(conceptEmbedding);
        _conceptDirections.Add(normalized);
    }

    /// <summary>
    /// Erases registered concepts from an embedding by projecting onto the null space.
    /// </summary>
    /// <param name="embedding">The embedding to modify.</param>
    /// <returns>The embedding with erased concepts.</returns>
    public Vector<T> EraseFromEmbedding(Vector<T> embedding)
    {
        var result = embedding.Clone();
        var strength = NumOps.FromDouble(_erasureStrength);

        foreach (var direction in _conceptDirections)
        {
            // Project embedding onto concept direction
            var dot = DotProduct(result, direction);
            var scaledDot = NumOps.Multiply(strength, dot);

            // Subtract projection: result = result - strength * (result . d) * d (d zero-padded to the embedding)
            result = Engine.Subtract(result, Engine.Multiply(DiffusionVectorOps.Fit(direction, result.Length), scaledDot));
        }

        return result;
    }

    /// <summary>
    /// Computes the concept presence score for an embedding (0 = absent, 1 = fully present).
    /// </summary>
    /// <param name="embedding">The embedding to evaluate.</param>
    /// <param name="conceptIndex">Index of the concept direction to check.</param>
    /// <returns>Cosine similarity with the concept direction.</returns>
    public double ComputeConceptPresence(Vector<T> embedding, int conceptIndex)
    {
        if (conceptIndex < 0 || conceptIndex >= _conceptDirections.Count)
        {
            return 0.0;
        }

        var direction = _conceptDirections[conceptIndex];
        var dot = NumOps.ToDouble(DotProduct(embedding, direction));
        var normEmb = VectorNorm(embedding);

        return normEmb > 1e-10 ? Math.Abs(dot) / normEmb : 0.0;
    }

    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        var sum = NumOps.Zero;
        var len = Math.Min(a.Length, b.Length);
        sum = NumOps.Add(sum, Engine.DotProduct(a, b));
        return sum;
    }

    private double VectorNorm(Vector<T> v)
    {
        return Engine.Norm(v.AsTensor().Cast<double>().ToVector());
    }

    private Vector<T> NormalizeVector(Vector<T> v)
    {
        var norm = VectorNorm(v);
        if (norm < 1e-10) return v;
        return Engine.TensorDivideScalar(v.AsTensor().Cast<double>(), norm).Cast<T>().ToVector();
    }

    /// <summary>
    /// Gets the number of registered concept directions.
    /// </summary>
    public int ConceptCount => _conceptDirections.Count;

    /// <summary>
    /// Gets the erasure strength.
    /// </summary>
    public double ErasureStrength => _erasureStrength;

    /// <summary>
    /// Gets whether general capability preservation is enabled.
    /// </summary>
    public bool PreserveGeneralCapability => _preserveGeneralCapability;
}
