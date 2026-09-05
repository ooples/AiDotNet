namespace AiDotNet.Enums;

/// <summary>Names one per-request chat setting so a reasoning-model profile can declare it unsupported.</summary>
/// <remarks>
/// <para>
/// Reasoning models reject some of the sampling controls that ordinary chat models accept. Sending
/// <c>temperature</c> or <c>top_p</c> to an o-series or GPT-5 model is answered with an HTTP 400, so the request
/// has to be adjusted before it leaves the process rather than after the provider rejects it. This enumeration is
/// the vocabulary a <c>ReasoningModelProfile</c> uses to say which settings must be removed, and the vocabulary a
/// diagnostic uses to say which setting it removed.
/// </para>
/// <para>
/// The members name the provider-neutral <c>ChatOptions</c> properties rather than the wire fields, because the
/// same profile has to drive several connectors whose wire names differ. The mapping to a wire field is the
/// connector's job.
/// </para>
/// <para><b>For Beginners:</b> When you ask an AI for a reply you can attach settings such as "be more creative"
/// (temperature) or "stop after 500 words" (maximum output tokens). Newer reasoning models do not accept all of
/// them. This list simply gives each setting a name so the library can report, in plain terms, which one it had to
/// leave out of your request and why.</para>
/// </remarks>
public enum ReasoningRequestParameter
{
    /// <summary>The sampling temperature (<c>ChatOptions.Temperature</c>, wire field <c>temperature</c>).</summary>
    Temperature = 0,

    /// <summary>Nucleus sampling mass (<c>ChatOptions.TopP</c>, wire field <c>top_p</c>).</summary>
    TopP = 1,

    /// <summary>Top-K sampling (<c>ChatOptions.TopK</c>); most OpenAI-shaped endpoints never send it.</summary>
    TopK = 2,

    /// <summary>The deterministic sampling seed (<c>ChatOptions.Seed</c>, wire field <c>seed</c>).</summary>
    Seed = 3,

    /// <summary>The stop sequences (<c>ChatOptions.StopSequences</c>, wire field <c>stop</c>).</summary>
    StopSequences = 4,

    /// <summary>The answer length cap (<c>ChatOptions.MaxOutputTokens</c>, wire field <c>max_tokens</c>).</summary>
    MaxOutputTokens = 5,

    /// <summary>The structured-output request (<c>ChatOptions.ResponseFormat</c>, wire field <c>response_format</c>).</summary>
    ResponseFormat = 6,

    /// <summary>The tool declarations (<c>ChatOptions.Tools</c>, wire fields <c>tools</c> and <c>tool_choice</c>).</summary>
    Tools = 7,

    /// <summary>The deliberation level a reasoning model accepts (wire field <c>reasoning_effort</c>).</summary>
    /// <remarks>
    /// Unlike the other members this one is never dropped: it is only ever added, because it has no counterpart on
    /// an ordinary chat model and is therefore never present in the request the caller configured.
    /// </remarks>
    ReasoningEffort = 8,

    /// <summary>The role name system messages are sent under (wire field <c>role</c> on the first message).</summary>
    SystemMessageRole = 9
}
