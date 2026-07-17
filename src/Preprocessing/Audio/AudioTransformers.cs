using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.Audio;

/// <summary>
/// A preprocessing step that runs a configured <see cref="IAudioEnhancer{T}"/> over audio-tensor inputs,
/// so audio enhancement composes with the rest of the preprocessing pipeline and is applied consistently to
/// both training and inference data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The pipeline input type; enhancement runs only when it is an audio <c>Tensor&lt;T&gt;</c>.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This lets you drop a noise-reducer (or dereverb / echo-canceller) into the same
/// preprocessing chain you use for scaling and imputing. Because it is a fitted pipeline step, the enhancer
/// learns its noise profile from your training audio and then cleans every batch — training and live — the same
/// way, instead of being a separate one-off pass that can treat train and test data differently.</para>
/// </remarks>
public sealed class AudioEnhancementTransformer<T, TInput> : IDataTransformer<T, TInput, TInput>
{
    private readonly IAudioEnhancer<T> _enhancer;

    /// <summary>Creates an enhancement pipeline step wrapping the given enhancer.</summary>
    /// <param name="enhancer">The configured audio enhancer to apply.</param>
    public AudioEnhancementTransformer(IAudioEnhancer<T> enhancer)
    {
        _enhancer = enhancer ?? throw new ArgumentNullException(nameof(enhancer));
    }

    /// <inheritdoc/>
    public bool IsFitted { get; private set; }

    /// <inheritdoc/>
    public int[]? ColumnIndices => null;

    /// <inheritdoc/>
    public bool SupportsInverseTransform => false;

    /// <inheritdoc/>
    public void Fit(TInput data)
    {
        // Learn the noise profile from the training audio so enhancement is data-consistent.
        if (data is Tensor<T> audio)
        {
            try { _enhancer.EstimateNoiseProfile(audio); }
            catch { /* enhancer does not use a noise profile; enhancement still applies in Transform. */ }
        }

        IsFitted = true;
    }

    /// <inheritdoc/>
    public TInput Transform(TInput data)
    {
        if (data is Tensor<T> audio)
        {
            return (TInput)(object)_enhancer.Enhance(audio);
        }

        return data; // non-audio input: identity.
    }

    /// <inheritdoc/>
    public TInput FitTransform(TInput data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc/>
    public TInput InverseTransform(TInput data)
        => throw new NotSupportedException("Audio enhancement is not invertible.");

    /// <inheritdoc/>
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
        => inputFeatureNames ?? Array.Empty<string>();
}

/// <summary>
/// A preprocessing step that runs a configured <see cref="IAudioEffect{T}"/> over audio-tensor inputs, so an
/// audio effect (reverb, EQ, compression, and the like) composes with the rest of the preprocessing pipeline —
/// typically as data augmentation applied consistently to training and inference audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The pipeline input type; the effect runs only when it is an audio <c>Tensor&lt;T&gt;</c>.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This drops an audio effect into your preprocessing chain. Effects are not fitted
/// from data, so this step just applies the effect's <c>Process</c> to each audio batch. Use it to colour or
/// augment audio the same way for every batch that flows through the pipeline.</para>
/// </remarks>
public sealed class AudioEffectTransformer<T, TInput> : IDataTransformer<T, TInput, TInput>
{
    private readonly IAudioEffect<T> _effect;

    /// <summary>Creates an effect pipeline step wrapping the given effect.</summary>
    /// <param name="effect">The configured audio effect to apply.</param>
    public AudioEffectTransformer(IAudioEffect<T> effect)
    {
        _effect = effect ?? throw new ArgumentNullException(nameof(effect));
    }

    /// <inheritdoc/>
    public bool IsFitted { get; private set; }

    /// <inheritdoc/>
    public int[]? ColumnIndices => null;

    /// <inheritdoc/>
    public bool SupportsInverseTransform => false;

    /// <inheritdoc/>
    public void Fit(TInput data) => IsFitted = true; // effects are configured, not fitted from data.

    /// <inheritdoc/>
    public TInput Transform(TInput data)
    {
        if (data is Tensor<T> audio)
        {
            return (TInput)(object)_effect.Process(audio);
        }

        return data; // non-audio input: identity.
    }

    /// <inheritdoc/>
    public TInput FitTransform(TInput data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc/>
    public TInput InverseTransform(TInput data)
        => throw new NotSupportedException("Audio effects are not invertible.");

    /// <inheritdoc/>
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
        => inputFeatureNames ?? Array.Empty<string>();
}
