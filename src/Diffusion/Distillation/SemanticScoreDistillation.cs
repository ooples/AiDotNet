using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Semantic Score Distillation (SemanticSDS) for semantically-guided 3D generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SemanticSDS enhances score distillation by incorporating semantic understanding from
/// vision-language models (e.g., CLIP). It decomposes the SDS gradient into semantic and
/// appearance components, allowing independent control over what the 3D object represents
/// versus how it looks. This reduces semantic drift and produces 3D objects that better
/// match the text prompt's meaning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes SDS produces 3D objects that look nice but don't match
/// the text description well (semantic drift). SemanticSDS splits the optimization into
/// "does it mean the right thing?" and "does it look good?", optimizing both separately.
/// This ensures the 3D model of a "red ceramic mug" actually looks like a mug, not just
/// a red blob.
/// </para>
/// <para>
/// Reference: Adapted from semantic-aware score distillation techniques combining CLIP
/// guidance with diffusion model scores for text-to-3D generation
/// </para>
/// </remarks>
public class SemanticScoreDistillation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _guidanceScale;
    private readonly double _semanticWeight;
    private readonly double _appearanceWeight;

    /// <summary>
    /// Gets the guidance scale.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the weight for the semantic component.
    /// </summary>
    public double SemanticWeight => _semanticWeight;

    /// <summary>
    /// Gets the weight for the appearance component.
    /// </summary>
    public double AppearanceWeight => _appearanceWeight;

    /// <summary>
    /// Initializes a new SemanticSDS instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model.</param>
    /// <param name="guidanceScale">CFG scale (default: 100.0).</param>
    /// <param name="semanticWeight">Weight for semantic alignment (default: 1.0).</param>
    /// <param name="appearanceWeight">Weight for appearance quality (default: 1.0).</param>
    public SemanticScoreDistillation(
        IDiffusionModel<T> diffusionModel,
        double guidanceScale = 100.0,
        double semanticWeight = 1.0,
        double appearanceWeight = 1.0)
    {
        _diffusionModel = diffusionModel;
        _guidanceScale = guidanceScale;
        _semanticWeight = semanticWeight;
        _appearanceWeight = appearanceWeight;
    }

    /// <summary>
    /// Computes the semantic SDS gradient with decomposed components.
    /// </summary>
    /// <param name="sdsGradient">Base SDS gradient (appearance component).</param>
    /// <param name="semanticGradient">CLIP/VL model semantic alignment gradient.</param>
    /// <returns>Combined semantic + appearance gradient.</returns>
    public Vector<T> ComputeGradient(Vector<T> sdsGradient, Vector<T> semanticGradient)
    {
        var gradient = new Vector<T>(sdsGradient.Length);
        var appWeight = NumOps.FromDouble(_appearanceWeight);
        var semWeight = NumOps.FromDouble(_semanticWeight);

        for (int i = 0; i < gradient.Length; i++)
        {
            var appGrad = NumOps.Multiply(appWeight, sdsGradient[i]);
            var semGrad = NumOps.Multiply(semWeight,
                i < semanticGradient.Length ? semanticGradient[i] : NumOps.Zero);
            gradient[i] = NumOps.Add(appGrad, semGrad);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the semantic alignment score between render embedding and text embedding.
    /// </summary>
    /// <param name="renderEmbedding">CLIP embedding of the rendered view.</param>
    /// <param name="textEmbedding">CLIP embedding of the text prompt.</param>
    /// <returns>Cosine similarity between embeddings.</returns>
    public T ComputeSemanticAlignment(Vector<T> renderEmbedding, Vector<T> textEmbedding)
    {
        var dotProduct = NumOps.Zero;
        var normRender = NumOps.Zero;
        var normText = NumOps.Zero;
        int len = Math.Min(renderEmbedding.Length, textEmbedding.Length);

        for (int i = 0; i < len; i++)
        {
            dotProduct = NumOps.Add(dotProduct,
                NumOps.Multiply(renderEmbedding[i], textEmbedding[i]));
            normRender = NumOps.Add(normRender,
                NumOps.Multiply(renderEmbedding[i], renderEmbedding[i]));
            normText = NumOps.Add(normText,
                NumOps.Multiply(textEmbedding[i], textEmbedding[i]));
        }

        var denominator = NumOps.Multiply(NumOps.Sqrt(normRender), NumOps.Sqrt(normText));
        if (NumOps.ToDouble(denominator) < 1e-8)
            return NumOps.Zero;

        return NumOps.Divide(dotProduct, denominator);
    }
}
