using System;

namespace AiDotNet.Models;

/// <summary>
/// Cloning for options classes.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> <c>options.Clone()</c> gives you a separate copy you can change without
/// affecting the original — useful for running the same model with one setting varied. You do not
/// have to write anything to make this work on your own options class: inherit from an options base
/// class and cloning is already correct.
/// </para>
/// <para>
/// Offered as an extension rather than a virtual method so that the return type is the caller's own
/// type. <c>myOptions.Clone()</c> yields <c>MyOptions</c>, not the abstract base, without every
/// options class having to override anything or the base class needing a self-referencing type
/// parameter that would show up in every derived signature.
/// </para>
/// </remarks>
public static class ModelOptionsCloneExtensions
{
    /// <summary>
    /// Creates an independent copy of an options instance.
    /// </summary>
    /// <typeparam name="T">The options type; inferred from <paramref name="source"/>.</typeparam>
    /// <param name="source">The options to copy.</param>
    /// <param name="options">
    /// Reserved for symmetry with model and layer cloning. Options hold configuration and nothing
    /// else, so every setting on <see cref="CloneOptions"/> that concerns learned state has nothing
    /// to act on here.
    /// </param>
    /// <returns>A new instance carrying the same configuration.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    /// <remarks>
    /// <para>
    /// Every property is carried, including those declared on base classes. That inherited surface
    /// is what a hand-written copy constructor cannot see from a type's own declarations, and it is
    /// where 71 of them silently dropped <c>ModelOptions.Seed</c> — a clone that kept the default
    /// seed while the original kept a configured one, changing results with nothing to show for it.
    /// </para>
    /// <para>
    /// Collections are duplicated rather than shared, so configuring the copy cannot reconfigure
    /// the original through a buffer they both point at.
    /// </para>
    /// </remarks>
    public static T Clone<T>(this T source, CloneOptions? options = null)
        where T : ModelOptions
    {
        if (source is null) throw new ArgumentNullException(nameof(source));

        _ = options;
        return (T)CloneEngine.CopyConfiguration(source);
    }
}
