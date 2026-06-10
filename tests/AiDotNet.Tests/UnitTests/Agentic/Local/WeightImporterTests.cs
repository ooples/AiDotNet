using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    public class WeightImporterTests
    {
        // A simple INamedTensorSource backed by a dictionary (stands in for a loaded safetensors/GGUF file).
        private sealed class DictionaryTensorSource : INamedTensorSource
        {
            private readonly Dictionary<string, double[]> _tensors;

            public DictionaryTensorSource(Dictionary<string, double[]> tensors) => _tensors = tensors;

            public IReadOnlyCollection<string> TensorNames => _tensors.Keys;

            public double[] ReadAsDouble(string name) => _tensors[name];
        }

        private static MambaLanguageModel<double> TinyModel()
        {
            var arch = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration, inputSize: 12, outputSize: 12);
            return new MambaLanguageModel<double>(arch, vocabSize: 12, modelDimension: 8, numLayers: 1, stateDimension: 4, maxSeqLength: 8);
        }

        [Fact(Timeout = 120000)]
        public async Task ImportsConcatenatedNamedTensors_IntoModelParameters()
        {
            var model = TinyModel();
            var count = (int)model.ParameterCount;

            // Split the parameter vector across two named tensors to exercise ordered concatenation.
            var half = count / 2;
            var source = new DictionaryTensorSource(new Dictionary<string, double[]>
            {
                ["a"] = Enumerable.Repeat(0.25, half).ToArray(),
                ["b"] = Enumerable.Repeat(0.75, count - half).ToArray(),
            });

            WeightImporter.ImportInto(model, new[] { "a", "b" }, source);

            var parameters = model.GetParameters();
            Assert.Equal(count, parameters.Length);
            Assert.Equal(0.25, parameters[0], 6);
            Assert.Equal(0.75, parameters[count - 1], 6);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 120000)]
        public async Task ParameterMap_NamesSegments_ByArchitecture()
        {
            // Mamba stack: EmbeddingLayer + MambaBlock x2 + LayerNorm + Dense head.
            var arch = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration, inputSize: 12, outputSize: 12);
            var model = new MambaLanguageModel<double>(arch, vocabSize: 12, modelDimension: 8, numLayers: 2, stateDimension: 4, maxSeqLength: 8);

            var segments = ModelParameterMap.Build(model);
            var names = segments.Select(s => s.Name).ToList();

            // Recognizable, architecture-derived names in flat-parameter order.
            Assert.Equal(new[] { "token_embd.0", "blk.0", "blk.1", "norm.0", "output.0" }, names);

            // Segments tile the flat vector exactly: contiguous, non-overlapping, summing to ParameterCount.
            var offset = 0;
            foreach (var segment in segments)
            {
                Assert.Equal(offset, segment.Offset);
                offset += segment.Length;
            }

            Assert.Equal((int)model.ParameterCount, offset);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 120000)]
        public async Task ImportByName_RoundTrips_ExportedWeights()
        {
            // Export a trained-ish model's weights by name, then import them by name into a freshly-initialized
            // model of the same architecture: the parameter vectors must match exactly. This proves the map's
            // ordering reconstructs the flat parameter vector.
            var source = TwoLayerModel();
            var sourceParams = source.GetParameters();

            // Perturb so the two models genuinely differ before import.
            var perturbed = new double[sourceParams.Length];
            for (var i = 0; i < perturbed.Length; i++)
            {
                perturbed[i] = Math.Sin(i * 0.5) * 0.3;
            }

            source.SetParameters(new Vector<double>(perturbed));

            var exported = WeightImporter.Export(source);
            Assert.Equal(new[] { "token_embd.0", "blk.0", "blk.1", "norm.0", "output.0" }, exported.Keys.ToArray());

            var target = TwoLayerModel();
            var tensorSource = new DictionaryTensorSource(new Dictionary<string, double[]>(exported));
            WeightImporter.ImportByName(target, tensorSource);

            var targetParams = target.GetParameters();
            Assert.Equal(source.ParameterCount, target.ParameterCount);
            for (var i = 0; i < targetParams.Length; i++)
            {
                Assert.Equal(perturbed[i], targetParams[i], 9);
            }

            await Task.CompletedTask;
        }

        private static MambaLanguageModel<double> TwoLayerModel()
        {
            var arch = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration, inputSize: 12, outputSize: 12);
            return new MambaLanguageModel<double>(arch, vocabSize: 12, modelDimension: 8, numLayers: 2, stateDimension: 4, maxSeqLength: 8);
        }

        [Fact(Timeout = 120000)]
        public async Task LengthMismatch_Throws()
        {
            var model = TinyModel();
            var source = new DictionaryTensorSource(new Dictionary<string, double[]>
            {
                ["only"] = new[] { 1.0, 2.0, 3.0 }, // far fewer than ParameterCount
            });

            Assert.Throws<InvalidOperationException>(() =>
                WeightImporter.ImportInto(model, new[] { "only" }, source));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task SafetensorsFile_IsANamedTensorSource()
        {
            // The readers implement INamedTensorSource, so a loaded file is directly importable.
            var data = new byte[8];
            BitConverter.GetBytes(1.0).CopyTo(data, 0);
            var header = @"{""w"":{""dtype"":""F64"",""shape"":[1],""data_offsets"":[0,8]}}";
            var headerBytes = System.Text.Encoding.UTF8.GetBytes(header);
            var buffer = new byte[8 + headerBytes.Length + data.Length];
            BitConverter.GetBytes((ulong)headerBytes.Length).CopyTo(buffer, 0);
            headerBytes.CopyTo(buffer, 8);
            data.CopyTo(buffer, 8 + headerBytes.Length);

            INamedTensorSource source = SafetensorsReader.Read(buffer);
            Assert.Contains("w", source.TensorNames);
            Assert.Equal(new[] { 1.0 }, source.ReadAsDouble("w"));
            await Task.CompletedTask;
        }
    }
}
