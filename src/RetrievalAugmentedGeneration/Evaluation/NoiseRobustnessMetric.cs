using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates the model's robustness to noisy or irrelevant documents in the context.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Tests how well the RAG system performs when irrelevant documents are included
/// in the retrieved context, measuring ability to filter signal from noise.
/// </remarks>
public class NoiseRobustnessMetric<T> : RAGMetricBase<T>
{
    private readonly T _noiseRatio;

    public override string Name => "Noise Robustness";
    public override string Description => "Evaluates robustness to noisy or irrelevant documents in the context";
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="NoiseRobustnessMetric{T}"/> class.
    /// </summary>
    /// <param name="noiseRatio">Ratio of noise documents to inject (0-1).</param>
    public NoiseRobustnessMetric(T noiseRatio)
    {
        _noiseRatio = noiseRatio;
    }

    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        throw new NotImplementedException("Noise robustness evaluation requires answer generation capability");
    }
}
