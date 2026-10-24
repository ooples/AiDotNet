using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for audio safety modules, judged on waveforms through <c>EvaluateAudio</c> (#2139).
/// </summary>
/// <remarks>
/// Audio safety modules were excluded from generation as if they wrapped a user-supplied inner model; they are built
/// from defaults alone. This base gives each one waveforms at <see cref="SampleRate"/>: a voiced tone with harmonics
/// and a little noise, and silence. Confidences must be probabilities, and the same waveform must get the same
/// answer.
/// </remarks>
public abstract class AudioSafetyModuleTestBase
{
    /// <summary>Subclasses return their concrete module.</summary>
    protected abstract IAudioSafetyModule<double> CreateModule();

    /// <summary>Sample rate of the test waveforms (the modules' default).</summary>
    protected virtual int SampleRate => 16000;

    /// <summary>Length of the test waveforms, in seconds.</summary>
    protected virtual double Seconds => 1.0;

    private IEnumerable<Vector<double>> Waveforms()
    {
        int n = (int)(SampleRate * Seconds);
        var rng = RandomHelper.CreateSeededRandom(2139);
        var voiced = new double[n];
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / SampleRate;
            voiced[i] = 0.4 * Math.Sin(2 * Math.PI * 140 * t) + 0.2 * Math.Sin(2 * Math.PI * 280 * t)
                + 0.1 * Math.Sin(2 * Math.PI * 420 * t) + 0.02 * (rng.NextDouble() * 2 - 1);
        }

        yield return new Vector<double>(voiced);
        yield return new Vector<double>(new double[n]);
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateAudio_ReturnsUnitIntervalConfidences()
    {
        await Task.Yield();
        var module = CreateModule();
        Assert.True(module.IsReady, $"Module '{module.ModuleName}' reports IsReady=false.");
        foreach (var audio in Waveforms())
        {
            var findings = module.EvaluateAudio(audio, SampleRate);
            Assert.NotNull(findings);
            foreach (var finding in findings)
                Assert.True(finding.Confidence >= 0.0 && finding.Confidence <= 1.0,
                    $"Confidence {finding.Confidence} for {finding.Category} is outside [0, 1]; it is a probability.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateAudio_IsDeterministic()
    {
        await Task.Yield();
        var module = CreateModule();
        foreach (var audio in Waveforms())
        {
            var first = module.EvaluateAudio(audio, SampleRate);
            var second = module.EvaluateAudio(audio, SampleRate);
            Assert.Equal(first.Count, second.Count);
            for (int i = 0; i < first.Count; i++)
            {
                Assert.Equal(first[i].Category, second[i].Category);
                Assert.Equal(first[i].Severity, second[i].Severity);
                Assert.Equal(first[i].Confidence, second[i].Confidence);
            }
        }
    }
}
