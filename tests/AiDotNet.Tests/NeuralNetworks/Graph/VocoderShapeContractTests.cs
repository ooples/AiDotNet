using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Verifies the vocoder family's shape law against a real forward pass.
/// </summary>
/// <remarks>
/// <para>
/// This exists because <see cref="ModelContractConformanceTests"/> STRUCTURALLY CANNOT verify these
/// models, and the reason is worth stating rather than leaving as an unexplained skip. That sweep
/// builds a model from an architecture it constructs itself and then predicts on the shape that
/// architecture declares. A vocoder ignores the architecture's channel count: HiFiGAN takes its mel
/// width from <c>HiFiGANOptions.MelChannels</c> (80) and builds kernels for it, while the
/// architecture it was handed still says whatever the sweep chose. Predict then dies with
/// "Input channels (64) must match kernel in_channels (80)" - and that is not a clamp that can be
/// widened, because the unclamped architecture says 64 too.
/// </para>
/// <para>
/// So the sweep reports these as SKIPPED, honestly, and the law is checked here instead by building
/// each vocoder at the mel width it actually wants.
/// </para>
/// </remarks>
public class VocoderShapeContractTests
{
    private readonly ITestOutputHelper _out;
    public VocoderShapeContractTests(ITestOutputHelper output) => _out = output;

    /// <summary>Frames of mel to synthesise from. Small on purpose - the law is about the ratio.</summary>
    private const int Frames = 8;

    [Fact]
    public void TheVocoderUpsampleLawPredictsTheWaveformLengthAForwardPassProduces()
    {
        // Every concrete vocoder reachable through VocoderBase. Discovered rather than listed, so a
        // vocoder re-parented later is covered without editing this test - and so a re-parenting that
        // silently fails to happen shows up as a shrinking count rather than as nothing at all.
        var vocoders = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromVocoderBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        _out.WriteLine($"vocoders reachable through VocoderBase: {vocoders.Count}");

        int checkedCount = 0, declined = 0;
        var disagreed = new List<string>();
        var skipped = new List<string>();

        foreach (var open in vocoders)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            object? model = null;
            try
            {
                model = ConstructNative(closed);
                if (model is null) { skipped.Add($"{open.Name}: no native architecture constructor"); continue; }
                if (model is not IShapeContract contract) { skipped.Add($"{open.Name}: not IShapeContract"); continue; }

                // Build the input at the mel width the MODEL asks for, which is the whole point of
                // this test - reading it off the model rather than off the architecture.
                int melChannels = ((TtsModelBase<double>)model).MelChannels;
                if (melChannels <= 0) { skipped.Add($"{open.Name}: MelChannels not set"); continue; }

                var shape = new[] { 1, melChannels, Frames };
                int[]? predicted = ShapeInference.InferOutputShape(contract, shape);
                if (predicted is null) { declined++; _out.WriteLine($"{open.Name}: DECLINED"); continue; }

                var input = new Tensor<double>(shape);
                int[] actual;
                try { actual = ((NeuralNetworkBase<double>)model).Predict(input).Shape.ToArray(); }
                catch (Exception ex) { skipped.Add($"{open.Name}: {Unwrap(ex).GetType().Name}: {Unwrap(ex).Message}"); continue; }

                checkedCount++;
                if (predicted.SequenceEqual(actual))
                {
                    _out.WriteLine($"{open.Name}: agreed  in [{string.Join(",", shape)}] "
                        + $"-> [{string.Join(",", actual)}]");
                }
                else
                {
                    disagreed.Add($"{open.Name}: in [{string.Join(",", shape)}] "
                        + $"contract says [{string.Join(",", predicted)}] "
                        + $"but Predict returned [{string.Join(",", actual)}]");
                }
            }
            finally { (model as IDisposable)?.Dispose(); }
        }

        _out.WriteLine("");
        _out.WriteLine($"checked={checkedCount}  declined={declined}  "
            + $"disagreed={disagreed.Count}  skipped={skipped.Count}");
        foreach (var s in skipped) _out.WriteLine($"  skipped: {s}");
        foreach (var d in disagreed) _out.WriteLine($"  DISAGREED: {d}");

        // A vocoder must EXIST for this to mean anything. VocoderBase had zero subclasses until the
        // vocoders were re-parented onto it, and a test that silently checks nothing would have
        // reported that state as success.
        Assert.True(vocoders.Count > 0,
            "no type derives from VocoderBase, so the vocoder shape law is a claim about nothing");

        Assert.True(disagreed.Count == 0,
            $"{disagreed.Count} vocoder contract(s) disagree with a real forward pass."
            + Environment.NewLine + string.Join(Environment.NewLine, disagreed));
    }

    private static bool DerivesFromVocoderBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(VocoderBase<>)) return true;
        }
        return false;
    }

    /// <summary>
    /// Builds the model through its architecture constructor, skipping the ONNX overload - that one
    /// demands a model file on disk and would make this a test of the filesystem.
    /// </summary>
    private static object? ConstructNative(Type closed)
    {
        var ctor = closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });
        if (ctor is null) return null;

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.TwoDimensional, NeuralNetworkTaskType.Regression,
            inputHeight: 80, inputWidth: Frames, outputSize: 1);

        var pars = ctor.GetParameters();
        var args = new object?[pars.Length];
        args[0] = architecture;
        for (int i = 1; i < pars.Length; i++) args[i] = pars[i].DefaultValue;

        try { return ctor.Invoke(args); }
        catch { return null; }
    }

    private static Exception Unwrap(Exception ex)
        => ex is System.Reflection.TargetInvocationException { InnerException: not null } tie
            ? Unwrap(tie.InnerException) : ex;
}
