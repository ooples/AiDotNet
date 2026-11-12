using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class FastSpeech2AcousticReferenceTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CompletePaperAcousticGraph_MatchesIndependentTorchObjectiveAndEveryParameterGradient(int variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", variant == 3 ? "adaspeech_acoustic_reference.json" : variant == 2 ? "fastspeech2_acoustic_reference.json" : "fastspeech_acoustic_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement values) => values.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => Values(reference.GetProperty("gradients").GetProperty(key));
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 5);
        using VarianceAdaptorTtsModelBase<double> model = variant == 1
            ? new FastSpeech<double>(architecture, new FastSpeechOptions
            {
                VocabSize = 12, EncoderDim = 8, HiddenDim = 8, NumHeads = 2,
                NumEncoderLayers = 1, NumDecoderLayers = 1, FftFilterSize = 16,
                DurationPredictorFilterSize = 8, DurationPredictorDropout = 0, DropoutRate = 0, MelChannels = 5
            })
            : variant == 3 ? new AdaSpeech<double>(architecture, new AdaSpeechOptions
            {
                VocabSize = 12, EncoderDim = 8, HiddenDim = 8, NumHeads = 2,
                NumEncoderLayers = 1, NumDecoderLayers = 1, FftFilterSize = 16,
                VariancePredictorFilterSize = 8, VariancePredictorDropout = 0, DropoutRate = 0, MelChannels = 5,
                AcousticConditionFilterSize = 8, AcousticConditionDropout = 0, NumSpeakers = 3
            }) : new FastSpeech2<double>(architecture, new FastSpeech2Options
        {
            VocabSize = 12, EncoderDim = 8, HiddenDim = 8, NumHeads = 2,
            NumEncoderLayers = 1, NumDecoderLayers = 1, FftFilterSize = 16,
            VariancePredictorFilterSize = 8, VariancePredictorDropout = 0, DropoutRate = 0, MelChannels = 5
        });
        model.DisableAutoStreaming();
        model.SetTrainingMode(false);
        // Materialize lazy parameters before assigning the independent reference state.
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 4 }, new Vector<double>(Values(reference.GetProperty("tokens")))),
            Mel = new Tensor<double>(new[] { 16, 5 }, new Vector<double>(Values(reference.GetProperty("target")))),
            Durations = reference.GetProperty("durations").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            Pitch = Values(reference.GetProperty("f0")), Energy = Values(reference.GetProperty("energy")), SpeakerId = 1
        };
        if (model is AdaSpeech<double> ada) ada.CurrentPhase = AdaSpeechTrainingPhase.SourceJoint;
        model.EvaluateTrainingObjective(sample);
        var expected = new Dictionary<Tensor<double>, double[]>();
        static T Field<T>(object owner, string name) => (T)owner.GetType().GetField(name,
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;
        static double[] Transpose(double[] values, int output, int input)
        {
            var transposed = new double[values.Length];
            for (int o = 0; o < output; o++) for (int i = 0; i < input; i++)
                transposed[i * output + o] = values[o * input + i];
            return transposed;
        }
        void Copy(Tensor<double> tensor, double[] weight, double[] gradient)
        {
            Assert.Equal(tensor.Length, weight.Length);
            for (int i = 0; i < weight.Length; i++) tensor.SetFlat(i, weight[i]);
            AiDotNetEngine.Current.InvalidatePersistentTensor(tensor);
            expected.Add(tensor, gradient);
        }
        void Embedding(EmbeddingLayer<double> layer, string name) =>
            Copy(Field<Tensor<double>>(layer, "_embeddingTensor"), Weight(name + ".weight"), Gradient(name + ".weight"));
        void Linear(DenseLayer<double> layer, string name, int input, int output)
        {
            TorchParameters.Linear(AiDotNetEngine.Current, layer, input, output, Weight(name + ".weight"), Weight(name + ".bias"));
            expected.Add(layer.GetWeights(), Transpose(Gradient(name + ".weight"), output, input));
            expected.Add(layer.GetBiases(), Gradient(name + ".bias"));
        }
        void Normalization(LayerNormalizationLayer<double> layer, string name)
        {
            TorchParameters.LayerNorm(AiDotNetEngine.Current, layer, Weight(name + ".weight"), Weight(name + ".bias"));
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Equal(2, owned.Count);
            expected.Add(owned[0], Gradient(name + ".weight")); expected.Add(owned[1], Gradient(name + ".bias"));
        }
        void ConditionalNormalization(ConditionalLayerNormalizationLayer<double> layer, string name)
        {
            foreach (var (projection, suffix) in new[] { (layer.ScaleProjection, ".scale.weight"), (layer.BiasProjection, ".shift.weight") })
            {
                TorchParameters.Linear(projection, 8, 8, Weight(name + suffix));
                var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { projection });
                Assert.Single(owned);
                expected.Add(owned[0], Transpose(Gradient(name + suffix), 8, 8));
            }
        }
        void Convolution(Conv1DLayer<double> layer, string name)
        {
            layer.SetParameters(new Vector<double>(Weight(name + ".weight").Concat(Weight(name + ".bias")).ToArray()));
            var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Equal(2, owned.Count);
            expected.Add(owned[0], Gradient(name + ".weight")); expected.Add(owned[1], Gradient(name + ".bias"));
        }
        void Transformer(FeedForwardTransformerBlock<double> block, string name)
        {
            var attention = Field<MultiHeadAttentionLayer<double>>(block, "_attention");
            var weight = Weight(name + ".attention.in_proj_weight");
            var gradient = Gradient(name + ".attention.in_proj_weight");
            Tensor<double>[] projections = [attention.GetQueryWeights(), attention.GetKeyWeights(), attention.GetValueWeights()];
            for (int i = 0; i < 3; i++) Copy(projections[i], Transpose(weight.Skip(i * 64).Take(64).ToArray(), 8, 8),
                Transpose(gradient.Skip(i * 64).Take(64).ToArray(), 8, 8));
            Copy(attention.GetOutputWeights(), Transpose(Weight(name + ".attention.out_proj.weight"), 8, 8),
                Transpose(Gradient(name + ".attention.out_proj.weight"), 8, 8));
            Copy(Field<Tensor<double>>(attention, "_outputBias"), Weight(name + ".attention.out_proj.bias"),
                Gradient(name + ".attention.out_proj.bias"));
            if (variant == 3 && name == "decoder")
            {
                ConditionalNormalization(Field<ConditionalLayerNormalizationLayer<double>>(block, "_attentionConditionalNorm"), name + ".norm1");
                ConditionalNormalization(Field<ConditionalLayerNormalizationLayer<double>>(block, "_convConditionalNorm"), name + ".norm2");
            }
            else
            {
                Normalization(Field<LayerNormalizationLayer<double>>(block, "_attentionNorm"), name + ".norm1");
                Normalization(Field<LayerNormalizationLayer<double>>(block, "_convNorm"), name + ".norm2");
            }
            Convolution(Field<Conv1DLayer<double>>(block, "_conv1"), name + ".conv1");
            Convolution(Field<Conv1DLayer<double>>(block, "_conv2"), name + ".conv2");
        }
        void Predictor(VariancePredictorLayer<double> layer, string name, int output)
        {
            Convolution(Field<Conv1DLayer<double>>(layer, "_conv1"), name + ".conv1");
            Convolution(Field<Conv1DLayer<double>>(layer, "_conv2"), name + ".conv2");
            Normalization(Field<LayerNormalizationLayer<double>>(layer, "_norm1"), name + ".norm1");
            Normalization(Field<LayerNormalizationLayer<double>>(layer, "_norm2"), name + ".norm2");
            Linear(Field<DenseLayer<double>>(layer, "_projection"), name + ".projection", 8, output);
        }
        Embedding((EmbeddingLayer<double>)model.Layers[0], "embedding");
        Transformer((FeedForwardTransformerBlock<double>)model.Layers[2], "encoder");
        var adaptor = model.VarianceAdaptor;
        Predictor(Field<VariancePredictorLayer<double>>(adaptor, "_durationPredictor"), "duration", 1);
        if (variant != 1)
        {
        Predictor(Field<VariancePredictorLayer<double>>(adaptor, "_pitchPredictor"), "pitch", 10);
        Linear(Field<DenseLayer<double>>(adaptor, "_pitchStatistics"), "statistics", 8, 2);
        Embedding(Field<EmbeddingLayer<double>>(adaptor, "_pitchEmbedding"), "pitch_embedding");
        Predictor(Field<VariancePredictorLayer<double>>(adaptor, "_energyPredictor"), "energy", 1);
        Embedding(Field<EmbeddingLayer<double>>(adaptor, "_energyEmbedding"), "energy_embedding");
        }
        Transformer((FeedForwardTransformerBlock<double>)model.Layers[5], "decoder");
        if (variant == 3)
        {
            ConditionalNormalization((ConditionalLayerNormalizationLayer<double>)model.Layers[6], "final_norm");
            Embedding(Field<EmbeddingLayer<double>>(model, "_speakerEmbedding"), "speaker");
            foreach (var (field, name) in new[] { ("_utteranceEncoder", "utterance"), ("_phonemeEncoder", "phoneme"), ("_phonemePredictor", "phoneme_predictor") })
            {
                var encoder = Field<AcousticConditionEncoderLayer<double>>(model, field);
                Convolution(Field<Conv1DLayer<double>>(encoder, "_conv1"), name + ".conv1");
                Convolution(Field<Conv1DLayer<double>>(encoder, "_conv2"), name + ".conv2");
                Normalization(Field<LayerNormalizationLayer<double>>(encoder, "_norm1"), name + ".norm1");
                Normalization(Field<LayerNormalizationLayer<double>>(encoder, "_norm2"), name + ".norm2");
                if (name != "utterance") Linear(Field<DenseLayer<double>>(encoder, "_projection"), name + ".projection", 8, 4);
            }
            Linear(Field<DenseLayer<double>>(model, "_phonemeProjection"), "phoneme_projection", 4, 8);
        }
        Linear((DenseLayer<double>)model.Layers[variant == 3 ? 7 : 6], "mel", 8, 5);
        var components = (List<LayerBase<double>>)typeof(TtsModelBase<double>).GetField("ComponentLayers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Concat(components.Cast<ILayer<double>>()).ToList());
        Assert.Equal(variant == 3 ? 96 : variant == 2 ? 63 : 39, parameters.Count); // Torch packs Q/K/V; native owns three matrices in each FFT block.
        Assert.Equal(parameters.Count, expected.Count);
        Assert.All(parameters, parameter => Assert.True(expected.ContainsKey(parameter), "An owned parameter lacks an independent gradient."));
        void Close(Tensor<double> actual, int output)
        {
            var wanted = Values(reference.GetProperty("outputs")[output]);
            Assert.Equal(wanted.Length, actual.Length);
            double error = actual.ToArray().Zip(wanted, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 1e-9, $"Complete acoustic graph output {output} differs by {error:R}.");
        }
        var hidden = sample.Tokens;
        for (int i = 0; i < 3; i++) hidden = model.Layers[i].Forward(hidden);
        AcousticConditioning<double>? conditioning = null;
        if (variant == 3)
        {
            conditioning = (AcousticConditioning<double>)typeof(AdaSpeech<double>).GetMethod("ConditionForTraining", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(model, new object[] { hidden, sample, sample.Mel!, sample.Durations! })!;
            hidden = conditioning.Hidden;
            Close(hidden, 7);
            Assert.True(Math.Abs(conditioning.AuxiliaryLoss![0] - reference.GetProperty("losses")[5].GetDouble()) < 1e-9);
        }
        var adapted = adaptor.Adapt(hidden, new VarianceTargets
        {
            Durations = sample.Durations, Pitch = Values(reference.GetProperty("continuous_f0")), Energy = sample.Energy
        });
        Close(adapted.LogDuration, 1);
        if (variant != 1)
        {
            Close(adapted.PitchSpectrogram!, 2); Close(adapted.PitchStatistics!, 3); Close(adapted.Energy!, 4);
        }
        var predictedMel = adapted.Expanded;
        for (int i = 4; i < model.Layers.Count; i++)
            predictedMel = model.Layers[i] is LayerBase<double> { RequiresMultipleInputs: true } conditioned
                ? conditioned.Forward(predictedMel, conditioning!.DecoderCondition!) : model.Layers[i].Forward(predictedMel);
        Close(predictedMel, 0);
        var build = typeof(VarianceAdaptorTtsModelBase<double>).GetMethod("BuildObjective", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var objective = ((Tensor<double> Tokens, Tensor<double> Mel, Func<Tensor<double>, Tensor<double>, Tensor<double>> Objective))build.Invoke(model, new object[] { sample })!;
        using var tape = new GradientTape<double>();
        var loss = objective.Objective(objective.Tokens, objective.Mel);
        Assert.True(Math.Abs(loss[0] - reference.GetProperty("loss").GetDouble()) < 1e-8,
            $"Complete paper objective differs: {loss[0]:R}.");
        var gradients = tape.ComputeGradients(loss, sources: parameters);
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.ContainsKey(parameter), "Missing full-graph gradient.");
            double error = gradients[parameter].ToArray().Zip(expected[parameter], (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 1e-8, $"Full-graph parameter gradient differs by {error:R}.");
        }
        Assert.True(Math.Abs(model.EvaluateTrainingObjective(sample) - loss[0]) < 1e-10);
    }
}
