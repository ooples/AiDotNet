using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Preprocessing;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Safety;

/// <summary>
/// Guards the single-call Vector input/output SafetyFilter fast paths in
/// <see cref="AiModelResult{T, TInput, TOutput}.Predict"/> (issue #1458 follow-through
/// to the #1447 matrix fast paths).
///
/// For the default numeric <see cref="AiDotNet.AdversarialRobustness.Safety.SafetyFilter{T}"/>,
/// the per-call <c>ValidateInput</c>/<c>FilterOutput</c> build a <c>ConvertToText</c>
/// string over every element (three times in <c>ValidateInput</c>) and run
/// English-phrase jailbreak/harmful-content regexes that can never match numeric
/// stringifications. The fast paths skip that meaningless text scan. These tests lock
/// the two invariants that make skipping safe:
///   1. For clean/finite numeric data the fast path produces bit-identical output to a
///      disabled filter (default filter never modifies numeric Vector I/O).
///   2. Non-finite / over-length input still throws (the input fast path falls through
///      to the full per-call validation for exact diagnostics).
/// and that a CUSTOM <see cref="ISafetyFilter{T}"/> is still consulted on both Vector
/// paths (only the concrete default filter is fast-pathed).
/// </summary>
public class AiModelResultVectorSafetyFastPathTests
{
    private const int Dim = 6;

    private static AiModelResult<double, Vector<double>, Vector<double>> BuildResult(
        SafetyFilterConfiguration<double>? safety)
    {
        var optimizationResult = new OptimizationResult<double, Vector<double>, Vector<double>>
        {
            BestSolution = new EchoVectorModel(Dim)
        };

        var options = new AiModelResultOptions<double, Vector<double>, Vector<double>>
        {
            OptimizationResult = optimizationResult,
            PreprocessingInfo = new PreprocessingInfo<double, Vector<double>, Vector<double>>(),
            SafetyFilterConfiguration = safety
        };

        return new AiModelResult<double, Vector<double>, Vector<double>>(options);
    }

    private static Vector<double> Finite()
    {
        var v = new Vector<double>(Dim);
        for (int i = 0; i < Dim; i++)
        {
            v[i] = (i + 1) * 1.5 - 2.0; // mix of positive/negative non-trivial values
        }
        return v;
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_VectorOutput_DefaultFilter_BitIdenticalToDisabledFilter()
    {
        var input = Finite();

        var withDefault = BuildResult(null); // null config => default SafetyFilter<T> (fast path)
        var disabled = BuildResult(new SafetyFilterConfiguration<double> { Enabled = false });

        var fast = withDefault.Predict(input);
        var baseline = disabled.Predict(input);

        Assert.Equal(baseline.Length, fast.Length);
        for (int i = 0; i < baseline.Length; i++)
        {
            // Identical bits: the default numeric filter must not alter the output.
            Assert.Equal(baseline[i], fast[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_VectorInput_DefaultFilter_FiniteInput_PassesThroughUnchanged()
    {
        var input = Finite();

        var withDefault = BuildResult(null);

        // EchoVectorModel returns a clone of its input, so a clean numeric input must
        // reach the model untouched and round-trip exactly through the input fast path.
        var output = withDefault.Predict(input);

        Assert.Equal(input.Length, output.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], output[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_VectorInput_DefaultFilter_NonFiniteInput_StillThrows()
    {
        var input = Finite();
        input[2] = double.NaN; // non-finite => fast path must fall through to full validation

        var withDefault = BuildResult(null);

        Assert.Throws<InvalidOperationException>(() => withDefault.Predict(input));
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_VectorOutput_CustomFilter_IsStillApplied()
    {
        var input = Finite();
        var custom = new RecordingSafetyFilter(zeroOutput: true);

        var result = BuildResult(new SafetyFilterConfiguration<double> { Filter = custom });

        var output = result.Predict(input);

        Assert.True(custom.FilterOutputCalls > 0,
            "Custom ISafetyFilter.FilterOutput must still be invoked on the Vector output path.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.Equal(0.0, output[i]); // custom filter zeroed the output, proving it ran
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_VectorInput_CustomFilter_IsStillConsulted()
    {
        var input = Finite();
        var custom = new RecordingSafetyFilter(zeroOutput: false);

        var result = BuildResult(new SafetyFilterConfiguration<double> { Filter = custom });

        _ = result.Predict(input);

        Assert.True(custom.ValidateInputCalls > 0,
            "Custom ISafetyFilter.ValidateInput must still be invoked on the Vector input path.");
    }

    /// <summary>Minimal Vector-&gt;Vector model that echoes a clone of its input.</summary>
    [ModelMetadataExempt]
    private sealed class EchoVectorModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _dim;
        private List<int> _activeFeatures;

        public EchoVectorModel(int dim)
        {
            _dim = dim;
            _activeFeatures = Enumerable.Range(0, dim).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i];
            }
            return output;
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new EchoVectorModel(_dim);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new EchoVectorModel(_dim);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(_dim);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(_dim);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new EchoVectorModel(_dim);
        public long ParameterCount => _dim;
        public bool SupportsParameterInitialization => true;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, _dim).ToDictionary(i => $"Feature{i}", i => 1.0 / _dim);
        public Vector<double> SanitizeParameters(Vector<double> parameters) => parameters;
        public void Dispose() { }
    }

    /// <summary>
    /// Custom <see cref="ISafetyFilter{T}"/> that records invocation counts so a test can
    /// prove the per-call path is still taken for non-default filters. Optionally zeroes
    /// the output to make the FilterOutput effect observable end-to-end.
    /// </summary>
    private sealed class RecordingSafetyFilter : ISafetyFilter<double>
    {
        private readonly bool _zeroOutput;
        private readonly SafetyFilterOptions<double> _options = new();

        public RecordingSafetyFilter(bool zeroOutput)
        {
            _zeroOutput = zeroOutput;
        }

        public int ValidateInputCalls { get; private set; }
        public int FilterOutputCalls { get; private set; }

        public SafetyValidationResult<double> ValidateInput(Vector<double> input)
        {
            ValidateInputCalls++;
            return new SafetyValidationResult<double> { IsValid = true, SafetyScore = 1.0 };
        }

        public SafetyFilterResult<double> FilterOutput(Vector<double> output)
        {
            FilterOutputCalls++;
            if (_zeroOutput)
            {
                var zeroed = new Vector<double>(output.Length);
                return new SafetyFilterResult<double>
                {
                    IsSafe = false,
                    SafetyScore = 0.0,
                    FilteredOutput = zeroed,
                    WasModified = true
                };
            }

            return new SafetyFilterResult<double>
            {
                IsSafe = true,
                SafetyScore = 1.0,
                FilteredOutput = output,
                WasModified = false
            };
        }

        public JailbreakDetectionResult<double> DetectJailbreak(Vector<double> input) => new();
        public HarmfulContentResult<double> IdentifyHarmfulContent(Vector<double> content) => new();
        public double ComputeSafetyScore(Vector<double> content) => 1.0;
        public SafetyFilterOptions<double> GetOptions() => _options;
        public void Reset() { }
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }
}
