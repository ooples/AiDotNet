using AiDotNet.HarmonicEngine.Activations;
using AiDotNet.HarmonicEngine.Benchmarks;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Experiment 2: Nonlinearity ablation study.
/// Tests that different spectral nonlinearities produce distinct, valid transformations
/// and evaluates their effectiveness for periodic signal separation.
/// </summary>
public class NonlinearityAblationTests
{
    private readonly ITestOutputHelper _output;

    public NonlinearityAblationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void ModReLU_PreservesSignForLargeInputs()
    {
        var activation = new ModReLUActivation<double>(bias: -0.1);

        // Large positive: should pass through
        Assert.True(activation.Activate(5.0) > 0);
        // Large negative: should pass through with same sign
        Assert.True(activation.Activate(-5.0) < 0);
        // Small input below threshold: should be zeroed
        Assert.Equal(0.0, activation.Activate(0.05), 2);
    }

    [Fact]
    public void SpectralGating_OutputBoundedByInput()
    {
        var activation = new SpectralGatingActivation<double>(weight: 5.0, bias: -1.0);

        // Gate is in [0,1], so |output| <= |input|
        for (double x = -3.0; x <= 3.0; x += 0.5)
        {
            double output = activation.Activate(x);
            Assert.True(Math.Abs(output) <= Math.Abs(x) + 1e-10,
                $"SpectralGating output {output} should not exceed input magnitude {Math.Abs(x)}");
        }
    }

    [Fact]
    public void InstantaneousFreq_VectorActivation_DiffersFromScalar()
    {
        var activation = new InstantaneousFreqActivation<double>(modulationStrength: 0.5);
        int n = 64;

        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * 5 * i / n);
        }

        // Vector activation uses Hilbert transform
        var vectorResult = activation.Activate(signal);

        // Scalar activation uses tanh fallback
        var scalarResult = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            scalarResult[i] = activation.Activate(signal[i]);
        }

        // They should be different (vector uses context, scalar doesn't)
        double diff = 0;
        for (int i = 0; i < n; i++)
        {
            diff += Math.Abs(vectorResult[i] - scalarResult[i]);
        }

        Assert.True(diff > 0.01,
            $"Vector and scalar activations should differ, but L1 diff = {diff}");

        _output.WriteLine($"Vector vs scalar L1 difference: {diff:F4}");
    }

    [Fact]
    public void AllActivations_ProduceNonzeroOutput()
    {
        int n = 64;
        var signal = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * 3 * i / n) + 0.5 * Math.Cos(2 * Math.PI * 7 * i / n);
        }

        var activations = new IActivationFunction<double>[]
        {
            new ModReLUActivation<double>(-0.1),
            new SpectralGatingActivation<double>(5.0, -1.0),
            new InstantaneousFreqActivation<double>(0.5),
        };

        foreach (var act in activations)
        {
            var output = act.Activate(signal);

            double energy = 0;
            for (int i = 0; i < n; i++)
            {
                energy += output[i] * output[i];
            }

            Assert.True(energy > 0.01,
                $"{act.GetType().Name} should produce non-zero output, energy = {energy}");

            _output.WriteLine($"{act.GetType().Name}: output energy = {energy:F4}");
        }
    }

    [Fact]
    public void AllActivations_ProduceDistinctIMDPatterns()
    {
        // Different nonlinearities should create different intermodulation patterns
        int numCarriers = 4;
        int fftSize = 256;
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(numCarriers, fftSize);
        var bus = new SpectralBus<double>(carriers, fftSize);
        var extractor = new IMDExtractor<double>(carriers, fftSize);

        var amplitudes = new Vector<double>(numCarriers);
        amplitudes[0] = 1.0; amplitudes[1] = 2.0; amplitudes[2] = 1.5; amplitudes[3] = 0.8;

        var encoded = bus.Encode(amplitudes);

        var activations = new IActivationFunction<double>[]
        {
            new ModReLUActivation<double>(-0.05),
            new SpectralGatingActivation<double>(3.0, -0.5),
            new InstantaneousFreqActivation<double>(0.3),
        };

        var imdMatrices = new List<Matrix<double>>();

        foreach (var act in activations)
        {
            var activated = act.Activate(encoded);
            var interactions = extractor.ExtractPairwise(activated);
            imdMatrices.Add(interactions);

            _output.WriteLine($"{act.GetType().Name} diagonal: [{interactions[0, 0]:F4}, {interactions[1, 1]:F4}, {interactions[2, 2]:F4}, {interactions[3, 3]:F4}]");
        }

        // Each pair of activations should produce different IMD patterns
        for (int a = 0; a < imdMatrices.Count; a++)
        {
            for (int b = a + 1; b < imdMatrices.Count; b++)
            {
                double diff = 0;
                for (int i = 0; i < numCarriers; i++)
                    for (int j = 0; j < numCarriers; j++)
                        diff += Math.Abs(imdMatrices[a][i, j] - imdMatrices[b][i, j]);

                Assert.True(diff > 0.001,
                    $"Activations {a} and {b} should produce different IMD patterns, diff = {diff}");
            }
        }
    }

    [Fact]
    public void AllActivations_DerivativesAreFinite()
    {
        var activations = new IActivationFunction<double>[]
        {
            new ModReLUActivation<double>(-0.1),
            new SpectralGatingActivation<double>(5.0, -1.0),
            new InstantaneousFreqActivation<double>(0.5),
        };

        double[] testInputs = { -2.0, -1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0, 2.0 };

        foreach (var act in activations)
        {
            foreach (var input in testInputs)
            {
                var deriv = act.Derivative(input);
                Assert.False(double.IsNaN(deriv),
                    $"{act.GetType().Name}.Derivative({input}) = NaN");
                Assert.False(double.IsInfinity(deriv),
                    $"{act.GetType().Name}.Derivative({input}) = Infinity");
            }
        }
    }
}
