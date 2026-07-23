namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// The kind of pretrained-model source a <see cref="PretrainedSource"/> points at.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This says <em>where</em> and <em>in what format</em> a pretrained
/// model lives. You rarely name it directly — the <see cref="PretrainedSource"/> factory methods
/// (<c>HuggingFace</c>, <c>Safetensors</c>, <c>Onnx</c>, <c>Gguf</c>) pick the right kind for you.
/// </para>
/// </remarks>
public enum PretrainedModelKind
{
    /// <summary>A Hugging Face Hub repository id (e.g. <c>meta-llama/Llama-3.1-8B-Instruct</c>),
    /// resolved by downloading <c>config.json</c> + the safetensors shards and caching them.</summary>
    HuggingFace,

    /// <summary>A local directory containing <c>config.json</c> and one or more
    /// <c>*.safetensors</c> files (optionally a <c>model.safetensors.index.json</c>).</summary>
    Safetensors,

    /// <summary>A local <c>.onnx</c> graph, run directly through the ONNX runtime.</summary>
    Onnx,

    /// <summary>A local <c>.gguf</c> checkpoint (llama.cpp format).</summary>
    Gguf,
}
