using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
public class ConceptEraser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
        var result = new Vector<T>(embedding.Length);
        for (int i = 0; i < embedding.Length; i++)
        {
            result[i] = embedding[i];
        }

        foreach (var direction in _conceptDirections)
        {
            // Project embedding onto concept direction
            var dot = DotProduct(result, direction);
            var scaledDot = NumOps.Multiply(NumOps.FromDouble(_erasureStrength), dot);

            // Subtract projection: result = result - strength * (result . d) * d
            for (int i = 0; i < result.Length && i < direction.Length; i++)
            {
                var projection = NumOps.Multiply(scaledDot, direction[i]);
                result[i] = NumOps.Subtract(result[i], projection);
            }
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
        for (int i = 0; i < len; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    private double VectorNorm(Vector<T> v)
    {
        double sumSq = 0;
        for (int i = 0; i < v.Length; i++)
        {
            var val = NumOps.ToDouble(v[i]);
            sumSq += val * val;
        }
        return Math.Sqrt(sumSq);
    }

    private Vector<T> NormalizeVector(Vector<T> v)
    {
        var norm = VectorNorm(v);
        if (norm < 1e-10) return v;

        var result = new Vector<T>(v.Length);
        var invNorm = NumOps.FromDouble(1.0 / norm);
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = NumOps.Multiply(invNorm, v[i]);
        }
        return result;
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
