using System.Reflection;
using System.Security.Cryptography;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.PyTorchParity;

/// <summary>Full-V1 numerical evidence from export_hifigan_reference.py; never a timed benchmark.</summary>
internal static class HifiGanReferenceVerifier
{
    private sealed class Model(int seed) : HiFiGAN<float>(
        new NeuralNetworkArchitecture<float>(inputFeatures: 80, outputSize: 1), new HiFiGANOptions { SamplingSeed = seed })
    {
        internal Tensor<float> GenerateReference(Tensor<float> mel) => Generate(mel);
        internal Tensor<float> DiscriminatorObjective(Tensor<float> real, Tensor<float> generated)
            => DiscriminatorLoss(real, generated);
        internal Tensor<float> GeneratorObjective(Tensor<float> mel, Tensor<float> real) => GeneratorLoss(mel, real, true);

        internal (Tensor<float> Adversarial, Tensor<float> Matching, Tensor<float> Mel, Tensor<float> Total)
            Components(Tensor<float> mel, Tensor<float> real)
        {
            var critics = Field<HiFiGanDiscriminators<float>>(this, typeof(HiFiGAN<float>), "_discriminators");
            var analysis = Field<DifferentiableMel<float>>(this, typeof(HiFiGAN<float>), "_lossMel");
            var generated = Flat(Generate(mel));
            var fake = critics.Forward(generated);
            List<(Tensor<float> Score, List<Tensor<float>> Features)> realOut;
            Tensor<float> realMel;
            using (new NoGradScope<float>())
            {
                realOut = critics.Forward(real);
                realMel = analysis.Forward(real);
            }
            var adversarial = Sum(fake.Select(f => LeastSquaresGenerator(f.Score)));
            var matching = Sum(Enumerable.Range(0, fake.Count).Select(k => FeatureMatching(realOut[k].Features, fake[k].Features)));
            var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(analysis.Forward(generated), realMel)));
            var total = Engine.TensorAdd(Engine.TensorAdd(adversarial, Engine.TensorMultiplyScalar(matching, 2f)),
                Engine.TensorMultiplyScalar(melLoss, 45f));
            return (adversarial, matching, melLoss, total);
        }
    }

    private sealed record Binding(string Key, NormedConv1DLayer<float> Layer);
    private enum ComparisonKind { Standard, LearnedState, SpectralState }
    private sealed record Check(string Name, bool Pass, bool Finite, int Elements,
        double MaxAbsoluteError, double RelativeL2, double Cosine, double NormRatio,
        double ReferenceMaxAbsoluteValue, string ComparisonKind);
    private sealed class GradientTotals
    {
        internal double ExpectedSquared, ActualSquared, ErrorSquared, Dot;
        internal int Elements;
        internal bool Finite = true;
    }
    private sealed record Tolerance(double Absolute, double Relative, double GradientRelative, double GradientCosine);

    private static T Field<T>(object owner, Type type, string name)
    {
        var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException(type.FullName, name);
        return field.GetValue(owner) is T value ? value
            : throw new InvalidDataException($"{type.Name}.{name} is not a non-null {typeof(T).Name}.");
    }

    private static Binding[] Bindings(Model model) => Bindings(model.GeneratorLayers, model.DiscriminatorLayers);

    private static Binding[] Bindings(IEnumerable<LayerBase<float>> generator, IEnumerable<LayerBase<float>> discriminator)
    {
        var result = new List<Binding>();
        void Add(IEnumerable<LayerBase<float>> layers, string group)
        {
            int index = 0;
            foreach (var layer in layers)
            {
                if (layer is not NormedConv1DLayer<float> conv)
                    throw new InvalidDataException($"Unexpected full-V1 layer: {layer.GetType().FullName}");
                result.Add(new Binding($"{group}/{index++:000}", conv));
            }
            int expected = group == "generator" ? 78 : 54;
            if (index != expected) throw new InvalidDataException($"{group} has {index} layers; full V1 requires {expected}.");
        }
        Add(generator, "generator");
        Add(discriminator, "discriminator");
        return result.ToArray();
    }

    private static IEnumerable<(string Role, Tensor<float> Tensor)> Fields(Binding binding, bool learnedOnly = false)
    {
        var layer = binding.Layer;
        Tensor<float> Value(string name) => Field<Tensor<float>>(layer, typeof(NormedConv1DLayer<float>), name);
        yield return ("direction", Value("_direction"));
        if (layer.Normalization == ConvolutionNormalization.Weight) yield return ("gain", Value("_gain"));
        if (layer.HasBias) yield return ("bias", Value("_bias"));
        if (!learnedOnly && layer.Normalization == ConvolutionNormalization.Spectral)
        {
            yield return ("u", Value("_u"));
            yield return ("v", Value("_v"));
        }
    }

    private static void Import(Binding[] bindings, TorchCheckpointFile source)
    {
        var expectedKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach (var binding in bindings)
            foreach (var (role, tensor) in Fields(binding))
            {
                string key = binding.Key + "/" + role;
                expectedKeys.Add(key);
                source.CopyTo(key, tensor.Shape.ToArray(), tensor);
                AiDotNetEngine.Current.InvalidatePersistentTensor(tensor);
            }
        if (!expectedKeys.SetEquals(source.Names)) throw new InvalidDataException("The reference state has missing or extra fields.");
    }
    // The benchmark consumes the same active404-field mapping as numerical verification.
    private static Binding[] BenchmarkBindings(HiFiGAN<float> model)
        => Bindings(Field<HiFiGanGenerator<float>>(model, typeof(HiFiGAN<float>), "_generator").Layers,
            Field<HiFiGanDiscriminators<float>>(model, typeof(HiFiGAN<float>), "_discriminators").Layers);

    internal static IEnumerable<(string Name, Tensor<float> Value)> SharedBenchmarkState(HiFiGAN<float> model)
        => BenchmarkBindings(model).SelectMany(binding => Fields(binding)
            .Select(field => (binding.Key + "/" + field.Role, field.Tensor)));

    internal static void ImportSharedBenchmarkState(HiFiGAN<float> model, string path)
    {
        using var source = TorchCheckpointFile.Open(path, strictAxes: true);
        Import(BenchmarkBindings(model), source);
    }



    private static Tensor<float> Read(TorchCheckpointFile source, string name)
    {
        var shape = source.ShapeOf(name)?.Select(d => checked((int)d)).ToArray()
            ?? throw new InvalidDataException($"No shape for {name}.");
        var result = new Tensor<float>(shape);
        source.CopyTo(name, shape, result);
        return result;
    }

    private static Check Compare(string name, Tensor<float> actual, TorchCheckpointFile source, string key,
        Tolerance tolerance, GradientTotals? total = null, ComparisonKind kind = ComparisonKind.Standard)
    {
        // The engine represents reductions as rank-zero tensors; the reference serializes each scalar as [1].
        // Only a stored [1] is that scalar form: the reader does not check axes for an empty requested shape, so any
        // other one-element shape that differs from the actual tensor's is rejected here.
        int[] shape = actual.Shape.ToArray();
        if (actual.Length == 1 && source.ShapeOf(key) is { } stored
            && !stored.Select(d => (long)d).SequenceEqual(shape.Select(d => (long)d)))
        {
            if (stored.Length != 1 || stored[0] != 1)
                throw new InvalidDataException($"Scalar reference '{key}' must be stored as [1], not [{string.Join(", ", stored)}].");
            shape = Array.Empty<int>();
        }
        var expected = source.Read(key, shape);
        return Compare(name, actual.ToArray(), expected, tolerance, total, kind);
    }

    private static Check Compare(string name, float[] actual, double[] expected, Tolerance tolerance,
        GradientTotals? total = null, ComparisonKind kind = ComparisonKind.Standard)
    {
        if (actual.Length != expected.Length) throw new InvalidDataException($"{name}: tensor lengths differ.");
        double errorSquared = 0, expectedSquared = 0, actualSquared = 0, dot = 0, worst = 0, scale = 0;
        bool finite = true;
        for (int i = 0; i < actual.Length; i++)
        {
            double a = actual[i], e = expected[i];
            if (!double.IsFinite(a) || !double.IsFinite(e)) { finite = false; continue; }
            double error = a - e;
            errorSquared += error * error;
            expectedSquared += e * e;
            actualSquared += a * a;
            dot += a * e;
            worst = Math.Max(worst, Math.Abs(error));
            scale = Math.Max(scale, Math.Abs(e));
        }
        double relative = Math.Sqrt(errorSquared / Math.Max(expectedSquared, 1e-30));
        double ratio = expectedSquared > 0 ? Math.Sqrt(actualSquared / expectedSquared) : actualSquared == 0 ? 1 : 0;
        double cosine = expectedSquared > 0 && actualSquared > 0 ? dot / Math.Sqrt(expectedSquared * actualSquared)
            : expectedSquared == actualSquared ? 1 : 0;
        // AdamW can amplify small gradient differences in individual weights. The eager/compiled
        // reference control calibrates learned state by its tensorwise relative error and cosine.
        bool pass = kind == ComparisonKind.LearnedState
            ? finite && relative <= tolerance.Relative && cosine >= tolerance.GradientCosine
            : finite && worst <= tolerance.Absolute + tolerance.Relative * scale
                && (relative <= tolerance.Relative || worst <= tolerance.Absolute);
        if (total is not null)
        {
            total.ExpectedSquared += expectedSquared; total.ActualSquared += actualSquared;
            total.ErrorSquared += errorSquared; total.Dot += dot; total.Elements += actual.Length; total.Finite &= finite;
        }
        return new Check(name, pass, finite, actual.Length, worst, relative, cosine, ratio, scale,
            kind switch { ComparisonKind.LearnedState => "learned_state", ComparisonKind.SpectralState => "spectral_state", _ => "standard" });
    }

    private static Check GradientSummary(string name, GradientTotals total, Tolerance tolerance)
    {
        double relative = Math.Sqrt(total.ErrorSquared / Math.Max(total.ExpectedSquared, 1e-30));
        double ratio = Math.Sqrt(total.ActualSquared / Math.Max(total.ExpectedSquared, 1e-30));
        double cosine = total.Dot / Math.Sqrt(Math.Max(total.ExpectedSquared * total.ActualSquared, 1e-60));
        bool pass = total.Finite && total.Elements > 0 && total.ExpectedSquared > 0
            && relative <= tolerance.GradientRelative && Math.Abs(ratio - 1) <= tolerance.GradientRelative
            && cosine >= tolerance.GradientCosine;
        return new Check(name, pass, total.Finite, total.Elements, 0, relative, cosine, ratio, 0, "gradient_summary");
    }

    private static void CompareGradients(Binding[] bindings, string family, IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients,
        TorchCheckpointFile source, Tolerance tolerance, List<Check> checks)
    {
        var total = new GradientTotals();
        foreach (var binding in bindings.Where(b => b.Key.StartsWith(family + "/", StringComparison.Ordinal)))
            foreach (var (role, parameter) in Fields(binding, learnedOnly: true))
            {
                if (!gradients.TryGetValue(parameter, out var gradient))
                    throw new InvalidDataException($"Missing actual gradient: {binding.Key}/{role}.");
                string key = binding.Key + "/" + role;
                checks.Add(Compare("isolated-gradient/" + key, gradient, source, key, tolerance, total));
            }
        checks.Add(GradientSummary("isolated-gradient-summary/" + family, total, tolerance));
    }

    private static void CompareState(Binding[] bindings, TorchCheckpointFile expected, string phase,
        Tolerance tolerance, List<Check> checks)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var binding in bindings)
            foreach (var (role, tensor) in Fields(binding))
            {
                string key = binding.Key + "/" + role;
                names.Add(key);
                var kind = role switch
                {
                    "direction" or "gain" or "bias" => ComparisonKind.LearnedState,
                    "u" or "v" => ComparisonKind.SpectralState,
                    _ => throw new InvalidDataException($"Unauthorized state comparison role: {role}."),
                };
                checks.Add(Compare(phase + "/" + key, tensor, expected, key, tolerance, kind: kind));
            }
        if (!names.SetEquals(expected.Names)) throw new InvalidDataException($"{phase}: missing/extra reference state fields.");
    }

    private static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    internal static int Run(string directory, string output, string? workloadDirectory = null)
    {
        var checks = new List<Check>();
        var errors = new List<string>();
        // One snapshot supplies both the protocol and the receipt's hash, so the receipt cannot name a manifest that
        // replaced this one while the verification ran.
        byte[] manifestBytes = File.ReadAllBytes(Path.Combine(directory, "manifest.json"));
        string manifestSha256 = Convert.ToHexString(SHA256.HashData(manifestBytes)).ToLowerInvariant();
        JObject manifest;
        using (var manifestReader = new StreamReader(new MemoryStream(manifestBytes)))
            manifest = JObject.Parse(manifestReader.ReadToEnd());
        string device = AiDotNetEngine.Current.SupportsGpu ? "cuda" : "cpu";
        var tolerance = device == "cuda" ? new Tolerance(2e-4, 3e-3, .03, .995)
            : new Tolerance(2e-5, 1e-3, .01, .999);
        int fused = 0, eager = 0;
        bool previous = TensorCodecOptions.Current.EnableCompilation;
        try
        {
            if ((string?)manifest["schema"] != "hifigan-v1-equivalence-v1" || (string?)manifest["device"] != device
                || (bool?)manifest["compiled"] != true
                || (int?)manifest["steps"] != 3 || (int?)manifest["crop_start_frame"] != 0)
                throw new InvalidDataException("The manifest schema, device, torch.compile or three-step fixed-crop protocol differs.");
            if (device == "cuda" && !AiDotNetEngine.Current.Name.Contains("CUDA", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("CUDA parity requires an actual CUDA backend.");
            var hashes = manifest["files_sha256"] as JObject ?? throw new InvalidDataException("Missing reference hashes.");
            string[] requiredFiles = ["initial.safetensors", "reference.safetensors", "isolated-gradients.safetensors",
                "step1.safetensors", "step2.safetensors", "step3.safetensors"];
            if (!new HashSet<string>(hashes.Properties().Select(p => p.Name), StringComparer.Ordinal).SetEquals(requiredFiles))
                throw new InvalidDataException("The manifest must hash every required reference file exactly once.");
            foreach (var file in hashes.Properties())
            {
                if (Path.GetFileName(file.Name) != file.Name) throw new InvalidDataException("Reference hash paths must be filenames.");
                if (Hash(Path.Combine(directory, file.Name)) != (string?)file.Value)
                    throw new InvalidDataException($"Reference hash differs: {file.Name}");
            }
            using var reference = TorchCheckpointFile.Open(Path.Combine(directory, "reference.safetensors"), strictAxes: true);
            HifiBenchmarkFixture.ValidateMatchedReference(workloadDirectory, directory, manifest, reference);

            using var initial = TorchCheckpointFile.Open(Path.Combine(directory, "initial.safetensors"), strictAxes: true);
            using var referenceGradients = TorchCheckpointFile.Open(Path.Combine(directory, "isolated-gradients.safetensors"), strictAxes: true);
            using var model = new Model((int?)manifest["seed"] ?? throw new InvalidDataException("Missing seed."));
            var bindings = Bindings(model);
            if (model.TrainableParameterCount != 84_660_721 || model.TrainableParameterCount != (long?)manifest["trainable_parameters"])
                throw new InvalidDataException("Full-V1 trainable parameter counts differ.");
            Import(bindings, initial);
            TensorCodecOptions.Current.EnableCompilation = false;
            var mel = Read(reference, "mel/0");
            var real = Read(reference, "audio/0");
            if (!mel.Shape.ToArray().SequenceEqual(new[] { 1, 80, 32 }) || real.Rank != 1 || real.Length != 8192)
                throw new InvalidDataException("The full-V1 reference must use one fixed 32-frame/8192-sample crop.");
            model.SetTrainingMode(false);
            using (new NoGradScope<float>())
            {
                checks.Add(Compare("generator/output0", model.GenerateReference(mel), reference, "generator/output0", tolerance));
                foreach (int count in new[] { 1, 8, 16 })
                {
                    var input = Read(reference, $"inference/mel/{count}");
                    if (!input.Shape.ToArray().SequenceEqual(new[] { count, 80, 256 }))
                        throw new InvalidDataException($"Inference batch {count} changes the full256-frame geometry.");
                    using var arena = TensorArena.Create();
                    using var scope = TensorScope.Begin();
                    checks.Add(Compare($"inference/output/{count}", model.GenerateReference(input), reference,
                        $"inference/output/{count}", tolerance));
                }
            }

            model.SetTrainingMode(true);
            Tensor<float> fake;
            using (new NoGradScope<float>()) fake = model.GenerateReference(mel);
            var discriminatorParameters = bindings.Where(b => b.Key.StartsWith("discriminator/", StringComparison.Ordinal))
                .SelectMany(b => Fields(b, learnedOnly: true).Select(f => f.Tensor)).ToArray();
            using (var tape = new GradientTape<float>())
            {
                var loss = model.DiscriminatorObjective(real, AiDotNetEngine.Current.Reshape(fake, new[] { fake.Length }));
                checks.Add(Compare("isolated/discriminator_loss", loss, reference, "isolated/discriminator_loss", tolerance));
                CompareGradients(bindings, "discriminator", tape.ComputeGradients(loss, sources: discriminatorParameters),
                    referenceGradients, tolerance, checks);
            }

            Import(bindings, initial);
            model.SetTrainingMode(true);
            using (new NoGradScope<float>())
            {
                var terms = model.Components(mel, real);
                checks.Add(Compare("isolated/adversarial", terms.Adversarial, reference, "isolated/adversarial", tolerance));
                checks.Add(Compare("isolated/feature_matching", terms.Matching, reference, "isolated/feature_matching", tolerance));
                checks.Add(Compare("isolated/mel_l1", terms.Mel, reference, "isolated/mel_l1", tolerance));
                checks.Add(Compare("isolated/components_total", terms.Total, reference, "isolated/generator_loss", tolerance));
            }
            Import(bindings, initial);
            var generatorParameters = bindings.Where(b => b.Key.StartsWith("generator/", StringComparison.Ordinal))
                .SelectMany(b => Fields(b, learnedOnly: true).Select(f => f.Tensor)).ToArray();
            using (var tape = new GradientTape<float>())
            {
                var loss = model.GeneratorObjective(mel, real);
                checks.Add(Compare("isolated/production_generator_loss", loss, reference, "isolated/generator_loss", tolerance));
                CompareGradients(bindings, "generator", tape.ComputeGradients(loss, sources: generatorParameters),
                    referenceGradients, tolerance, checks);
            }

            Import(bindings, initial);
            TensorCodecOptions.Current.EnableCompilation = true;
            for (int step = 0; step < 3; step++)
            {
                var stepMel = Read(reference, $"mel/{step}");
                var stepAudio = Read(reference, $"audio/{step}");
                if (!stepMel.Shape.ToArray().SequenceEqual(new[] { 1, 80, 32 }) || stepAudio.Rank != 1 || stepAudio.Length != 8192)
                    throw new InvalidDataException($"Step {step} changes the fixed crop geometry.");
                model.Train(stepMel, stepAudio);
                checks.Add(Compare($"train/generator_loss/{step}", new[] { model.GetLastLoss() },
                    reference.Read($"train/generator_loss/{step}", new[] { 1 }), tolerance));
                using var state = TorchCheckpointFile.Open(Path.Combine(directory, $"step{step + 1}.safetensors"), strictAxes: true);
                CompareState(bindings, state, $"train/step{step + 1}", tolerance, checks);
            }
            fused = model.CustomFusedStepCount;
            eager = model.CustomEagerStepCount;
            if (fused != 6 || eager != 0 || !model.ObjectiveSessionCommitted)
                throw new InvalidOperationException($"The measured path did not stay fused: fused={fused}, eager={eager}.");
        }
        catch (Exception ex) { errors.Add(ex.ToString()); }
        finally { TensorCodecOptions.Current.EnableCompilation = previous; }
        if (checks.Count != 1615 || checks.Count(check => check.ComparisonKind == "learned_state") != 1164
            || checks.Count(check => check.ComparisonKind == "spectral_state") != 48
            || checks.Count(check => check.ComparisonKind != "learned_state") != 451)
            errors.Add("The receipt requires 1164 learned-state checks and 451 unchanged checks, including 48 spectral-state checks.");
        bool pass = errors.Count == 0 && checks.All(check => check.Pass);
        var receipt = new
        {
            schema = "hifigan-v1-equivalence-receipt-v1",
            model = "hifigan-v1",
            device,
            pass,
            reference_manifest_sha256 = manifestSha256,
            engine = AiDotNetEngine.Current.Name,
            tolerance,
            fused_steps = fused,
            eager_steps = eager,
            comparison_policy = new
            {
                schema = "hifigan-v1-calibrated-learned-state-v1",
                learned_state_checks = 1164,
                spectral_state_checks = 48,
                unchanged_checks = 451,
                learned_state = new
                {
                    finite_required = true,
                    relative_l2_max = tolerance.Relative,
                    cosine_min = tolerance.GradientCosine,
                    max_absolute_error_gate = false
                },
                other_criteria = "unchanged_original",
            },
            checks,
            errors,
            assemblies_sha256 = new Dictionary<string, string>
            {
                ["AiDotNet"] = Hash(typeof(HiFiGAN<float>).Assembly.Location),
                ["Tensors"] = Hash(typeof(Tensor<float>).Assembly.Location),
            },
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)) ?? ".");
        var settings = new JsonSerializerSettings
        { ContractResolver = new DefaultContractResolver { NamingStrategy = new SnakeCaseNamingStrategy() } };
        File.WriteAllText(output, JsonConvert.SerializeObject(receipt, Formatting.Indented, settings) + Environment.NewLine);
        Console.WriteLine($"HiFi-GAN full-V1 equivalence: {(pass ? "PASS" : "FAIL")}; {checks.Count} tensor checks; {errors.Count} errors; {output}");
        return pass ? 0 : 1;
    }
}
