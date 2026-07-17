using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// A model (or model wrapper) that carries its own tokenizer. When a served model implements this, the facade
/// resolves the tokenizer automatically, so <c>model.Generate("text")</c> and <c>model.Serve()</c> need no
/// explicitly-passed tokenizer — the zero-configuration path.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a model that already knows how to turn text into tokens (and back) can hand that
/// knowledge to the serving layer. If yours does, you don't have to pass a tokenizer when you generate or
/// serve — the library finds it.</para>
/// </remarks>
public interface IProvidesGenerationTokenizer
{
    /// <summary>Returns the tokenizer to use for text generation with this model.</summary>
    IGenerationTokenizer GetGenerationTokenizer();
}
