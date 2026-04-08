using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies which stage(s) of an AI pipeline a component operates in.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells you where in the data processing pipeline a component fits.
/// A retriever operates in the Retrieval stage, a chunker in DataIngestion, a reranker in
/// PostRetrieval. Some components span multiple stages.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ComponentType(ComponentType.Reranker)]
/// [PipelineStage(PipelineStage.PostRetrieval)]
/// public class CrossEncoderReranker&lt;T&gt; : RerankerBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class PipelineStageAttribute : Attribute
{
    /// <summary>
    /// Gets the pipeline stage this component operates in.
    /// </summary>
    public PipelineStage Stage { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PipelineStageAttribute"/> class.
    /// </summary>
    /// <param name="stage">The pipeline stage for this component.</param>
    public PipelineStageAttribute(PipelineStage stage)
    {
        Stage = stage;
    }
}
