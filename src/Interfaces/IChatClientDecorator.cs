using AiDotNet.Agentic.Models;

namespace AiDotNet.Interfaces;

/// <summary>A chat client that wraps another one and can hand out the client it wraps.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// A production chat client is usually a stack: retry around telemetry around content filtering around the client
/// that actually calls a model. That is invisible to callers who only send messages, and it should be. It is not
/// invisible to a caller that needs a capability of the innermost client rather than of the stack, such as asking
/// every member of an ensemble to score a candidate instead of asking the one member the ensemble would have picked.
/// Without a way to look through the wrappers, such a caller sees only the outermost object, silently finds no
/// ensemble, and quietly does something less useful than what it was configured to do.
/// </para>
/// <para><b>For Beginners:</b> Think of the layers as boxes inside boxes. This lets code that needs the thing in the
/// middle open the boxes, instead of giving up because the outside box is the wrong shape.</para>
/// </remarks>
public interface IChatClientDecorator<T> : IChatClient<T>
{
    /// <summary>Gets the client this one wraps.</summary>
    IChatClient<T> Inner { get; }
}
