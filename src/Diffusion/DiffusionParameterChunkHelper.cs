using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion;

/// <summary>
/// Shared helper for the buffer-and-delegate write path used by <c>SetParameterChunks</c> across the
/// diffusion model hierarchy.
/// </summary>
/// <remarks>
/// <para>
/// The "buffer a chunk stream, sum its lengths, fill one flat <see cref="Vector{T}"/>, then hand it to
/// <c>SetParameters</c>" logic was copy-pasted in <c>DiffusionModelBase.SetParameterChunks</c>,
/// <c>VAEModelBase.SetParameterChunks</c>, <c>NoisePredictorBase</c>'s legacy reflection fallback, and
/// every composite latent-diffusion model that streams a non-standard sub-module layout. Centralizing it
/// here gives a single source of truth so a change to the framing/validation rules lands everywhere at
/// once (the maintainability point raised on VAEModelBase.SetParameterChunks).
/// </para>
/// <para>
/// <b>For Beginners:</b> "chunks" are the model's weights handed over one tensor at a time (so a giant
/// model never has to build one enormous array in memory). This helper stitches those pieces back into a
/// single flat list of numbers and gives it to <c>SetParameters</c>. It is the bounded, back-compatible
/// write path for models whose full weight vector fits comfortably in memory (VAEs, composite wrappers).
/// The billion-parameter noise predictors deliberately do NOT use it — they copy one chunk per weight
/// tensor in place to stay fully streaming.
/// </para>
/// </remarks>
internal static class DiffusionParameterChunkHelper
{
    /// <summary>
    /// Buffers <paramref name="chunks"/> into one contiguous flat <see cref="Vector{T}"/> in stream
    /// order, rejecting a null sequence or any null element. Callers pass the result to their own
    /// <c>SetParameters</c> (and are responsible for any copy-on-write detach beforehand).
    /// </summary>
    /// <typeparam name="T">The numeric type of the model's parameters.</typeparam>
    /// <param name="chunks">The per-tensor parameter chunks, in <c>GetParameterChunks</c> order.</param>
    /// <returns>A single flat vector holding every chunk's values, concatenated in order.</returns>
    /// <exception cref="ArgumentNullException">The <paramref name="chunks"/> sequence is null.</exception>
    /// <exception cref="ArgumentException">The sequence contains a null tensor.</exception>
    internal static Vector<T> BufferToFlatVector<T>(IEnumerable<Tensor<T>> chunks)
    {
        if (chunks is null) throw new ArgumentNullException(nameof(chunks));

        var buffered = new List<Tensor<T>>();
        long total = 0;
        foreach (var chunk in chunks)
        {
            if (chunk is null)
                throw new ArgumentException("Chunk sequence contains a null tensor.", nameof(chunks));
            buffered.Add(chunk);
            total += chunk.Length;
        }

        var flat = new Vector<T>(checked((int)total));
        int offset = 0;
        foreach (var chunk in buffered)
        {
            var v = chunk.ToVector();
            for (int i = 0; i < v.Length; i++) flat[offset++] = v[i];
        }

        return flat;
    }
}
