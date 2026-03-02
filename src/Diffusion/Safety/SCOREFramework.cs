using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Safety;

/// <summary>
/// SCORE: Selective Concept Obliteration for Responsible Editing in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SCORE provides a structured framework for selectively removing unsafe or undesirable
/// concepts from diffusion models while preserving the model's overall generation quality.
/// It operates by fine-tuning cross-attention layers to redirect concept associations,
/// effectively "remapping" the erased concept to a neutral alternative.
/// </para>
/// <para>
/// <b>For Beginners:</b> SCORE works like a content filter built into the model itself.
/// Instead of blocking outputs after generation, it changes the model so it can't
/// generate certain content in the first place. When the model encounters a prompt for
/// erased content, it automatically redirects to generating something neutral instead.
/// </para>
/// <para>
/// Reference: Lu et al., "SCORE: Selective Concept Obliteration for Responsible Editing", 2024
/// </para>
/// </remarks>
public class SCOREFramework<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Dictionary<string, string> _conceptRemappings;
    private readonly double _remappingStrength;
    private readonly double _preservationWeight;

    /// <summary>
    /// Initializes a new SCORE framework instance.
    /// </summary>
    /// <param name="remappingStrength">Strength of concept remapping (default: 1.0).</param>
    /// <param name="preservationWeight">Weight for preserving non-target concepts (default: 0.1).</param>
    public SCOREFramework(
        double remappingStrength = 1.0,
        double preservationWeight = 0.1)
    {
        _conceptRemappings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        _remappingStrength = remappingStrength;
        _preservationWeight = preservationWeight;
    }

    /// <summary>
    /// Registers a concept to be remapped to a neutral alternative.
    /// </summary>
    /// <param name="targetConcept">The concept to erase.</param>
    /// <param name="neutralReplacement">The neutral concept to redirect to.</param>
    public void AddConceptRemapping(string targetConcept, string neutralReplacement)
    {
        _conceptRemappings[targetConcept] = neutralReplacement;
    }

    /// <summary>
    /// Computes the SCORE training loss for cross-attention remapping.
    /// </summary>
    /// <param name="originalAttention">Attention weights with the original concept.</param>
    /// <param name="remappedAttention">Attention weights with the neutral replacement.</param>
    /// <param name="preservedAttention">Attention weights for an unrelated concept (preservation target).</param>
    /// <param name="originalPreserved">Original attention weights for the preserved concept.</param>
    /// <returns>The combined remapping and preservation loss.</returns>
    public T ComputeSCORELoss(
        Vector<T> originalAttention,
        Vector<T> remappedAttention,
        Vector<T> preservedAttention,
        Vector<T> originalPreserved)
    {
        // Remapping loss: MSE between original concept attention and neutral replacement attention
        var remapLoss = ComputeMSE(originalAttention, remappedAttention);

        // Preservation loss: MSE between preserved concept attention before and after editing
        var preserveLoss = ComputeMSE(preservedAttention, originalPreserved);

        // Combined: remap_loss + preservation_weight * preserve_loss
        var scaledPreserve = NumOps.Multiply(NumOps.FromDouble(_preservationWeight), preserveLoss);
        var scaledRemap = NumOps.Multiply(NumOps.FromDouble(_remappingStrength), remapLoss);

        return NumOps.Add(scaledRemap, scaledPreserve);
    }

    /// <summary>
    /// Checks whether a concept has a registered remapping.
    /// </summary>
    /// <param name="concept">The concept to check.</param>
    /// <returns>True if the concept is registered for erasure.</returns>
    public bool IsConceptErased(string concept)
    {
        return _conceptRemappings.ContainsKey(concept);
    }

    /// <summary>
    /// Gets the neutral replacement for an erased concept.
    /// </summary>
    /// <param name="concept">The erased concept.</param>
    /// <returns>The neutral replacement, or null if concept is not erased.</returns>
    public string? GetReplacement(string concept)
    {
        return _conceptRemappings.TryGetValue(concept, out var replacement) ? replacement : null;
    }

    private T ComputeMSE(Vector<T> a, Vector<T> b)
    {
        var sum = NumOps.Zero;
        var len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return len > 0 ? NumOps.Divide(sum, NumOps.FromDouble(len)) : NumOps.Zero;
    }

    /// <summary>
    /// Gets the number of registered concept remappings.
    /// </summary>
    public int RemappingCount => _conceptRemappings.Count;

    /// <summary>
    /// Gets the remapping strength.
    /// </summary>
    public double RemappingStrength => _remappingStrength;

    /// <summary>
    /// Gets the preservation weight.
    /// </summary>
    public double PreservationWeight => _preservationWeight;
}
