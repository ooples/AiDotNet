using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.Reasoning.Benchmarks;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.ProgramSynthesis.Tooling;

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintHelp();
            return 0;
        }

        try
        {
            return args[0].ToLowerInvariant() switch
            {
                "train" => await RunTrainAsync(args.Skip(1).ToArray()).ConfigureAwait(false),
                "evaluate" => await RunEvaluateAsync(args.Skip(1).ToArray()).ConfigureAwait(false),
                _ => throw new ArgumentException($"Unknown command '{args[0]}'. Use --help for usage.")
            };
        }
        catch (OperationCanceledException)
        {
            Console.Error.WriteLine("Cancelled.");
            return 2;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
AiDotNet.ProgramSynthesis.Tooling

Usage:
  dotnet run --project tools/AiDotNet.ProgramSynthesis.Tooling -- train --model CodeT5 --language Python --train <train.jsonl> --output <model.model>
  dotnet run --project tools/AiDotNet.ProgramSynthesis.Tooling -- evaluate --model <model.model> --codeXGlue <codexglue.jsonl> --language Python --report <report.json>

Commands:
  train
    --train <path>              JSONL training set (required)
    --model-arch <CodeT5|CodeBERT|GraphCodeBERT>  (default: CodeT5)
    --language <ProgramLanguage> (default: Generic)
    --task <CodeTask>           (default: Generation)
    --epochs <int>              (default: 1)
    --max-samples <int>         (optional)
    --seed <int>                (default: 1337)
    --output <path>             (required)

  evaluate
    --model <path>              .model file (required)
    --humaneval <path>          HumanEval JSONL path (optional)
    --passk <int>               pass@k for HumanEval (default: 1)
    --codeXGlue <path>          CodeXGLUE-style JSONL path (optional)
    --language <ProgramLanguage> (default: Generic)
    --sample-size <int>         (optional)
    --report <path>             JSON report output (required)

Notes:
  - This tooling does not ship datasets or weights. Provide dataset paths locally.
  - HumanEval execution is performed via AiDotNet.Serving by default (recommended). Local execution requires setting:
      AIDOTNET_HUMANEVAL_EXECUTION=1
""");
    }

    private static async Task<int> RunTrainAsync(string[] args)
    {
        var trainPath = GetArgValue(args, "--train") ?? throw new ArgumentException("--train is required.");
        var outputPath = GetArgValue(args, "--output") ?? throw new ArgumentException("--output is required.");

        var modelArch = ParseEnum(GetArgValue(args, "--model-arch"), CodeModelArch.CodeT5);
        var language = ParseEnum(GetArgValue(args, "--language"), ProgramLanguage.Generic);
        var task = ParseEnum(GetArgValue(args, "--task"), CodeTask.Generation);
        var epochs = ParseInt(GetArgValue(args, "--epochs"), 1);
        var maxSamples = ParseNullableInt(GetArgValue(args, "--max-samples"));
        var seed = ParseInt(GetArgValue(args, "--seed"), 1337);

        var samples = ReadTrainingSamples(trainPath, maxSamples);
        if (samples.Count == 0)
        {
            throw new InvalidOperationException($"No training samples found in '{trainPath}'.");
        }

        var tokenizer = ProgramSynthesisTokenizerFactory.CreateDefault(language, splitIdentifiers: true);
        var architecture = CreateDefaultArchitecture<float>(modelArch, language, task, tokenizer);
        var model = CreateModel<float>(modelArch, architecture, tokenizer);

        var seededRandom = RandomHelper.CreateSeededRandom(seed);

        Console.WriteLine($"Training {modelArch} ({language}, {task}) on {samples.Count} samples (epochs={epochs})...");
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            ShuffleInPlace(samples, seededRandom);

            for (int i = 0; i < samples.Count; i++)
            {
                var sample = samples[i];
                var input = EncodeToTokenIdTensor<float>(tokenizer, sample.Prompt, architecture.MaxSequenceLength);
                var expected = EncodeToTokenIdTensor<float>(tokenizer, sample.Target, architecture.MaxSequenceLength);
                model.Train(input, expected);

                if ((i + 1) % 50 == 0 || i + 1 == samples.Count)
                {
                    Console.WriteLine($"Epoch {epoch}/{epochs}: {i + 1}/{samples.Count}");
                }
            }
        }

        var optimizationResult = new OptimizationResult<float, Tensor<float>, Tensor<float>>
        {
            BestSolution = model
        };

        var options = new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<float, Tensor<float>, Tensor<float>>(),
            Tokenizer = tokenizer
        };

        var result = new AiModelResult<float, Tensor<float>, Tensor<float>>(options);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        result.SaveModel(outputPath);
        Console.WriteLine($"Saved model to {outputPath}");
        return 0;
    }

    private static async Task<int> RunEvaluateAsync(string[] args)
    {
        var modelPath = GetArgValue(args, "--model") ?? throw new ArgumentException("--model is required.");
        var reportPath = GetArgValue(args, "--report") ?? throw new ArgumentException("--report is required.");

        var humanEvalPath = GetArgValue(args, "--humaneval");
        var passK = ParseInt(GetArgValue(args, "--passk"), 1);
        var codeXGluePath = GetArgValue(args, "--codeXGlue");
        var language = ParseEnum(GetArgValue(args, "--language"), ProgramLanguage.Generic);
        var sampleSize = ParseNullableInt(GetArgValue(args, "--sample-size"));

        var modelResult = LoadProgramSynthesisModel(modelPath);

        BenchmarkResult<float>? humanEval = null;
        if (!string.IsNullOrWhiteSpace(humanEvalPath))
        {
            if (!File.Exists(humanEvalPath))
            {
                throw new FileNotFoundException("HumanEval dataset not found.", humanEvalPath);
            }

            Environment.SetEnvironmentVariable("AIDOTNET_HUMANEVAL_DATASET", humanEvalPath);
            humanEval = await modelResult.EvaluateHumanEvalPassAtKAsync(passK, sampleSize, CancellationToken.None).ConfigureAwait(false);
        }

        BenchmarkResult<float>? codeXGlue = null;
        if (!string.IsNullOrWhiteSpace(codeXGluePath))
        {
            var options = new CodeXGlueBenchmarkOptions
            {
                DatasetFilePath = codeXGluePath,
                TaskName = "external",
                SourceField = "source",
                TargetField = "target",
                IdField = "id",
                CategoryField = "category"
            };

            var benchmark = new CodeXGlueBenchmark<float>(options);
            codeXGlue = await benchmark.EvaluateAsync(
                prompt => Task.FromResult(
                    modelResult.GenerateCode(new CodeGenerationRequest
                    {
                        Language = language,
                        Description = prompt
                    }).GeneratedCode ?? string.Empty),
                sampleSize,
                CancellationToken.None).ConfigureAwait(false);
        }

        var report = new
        {
            GeneratedAtUtc = DateTimeOffset.UtcNow,
            ModelPath = modelPath,
            HumanEval = humanEval,
            CodeXGlue = codeXGlue
        };

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(reportPath))!);
        File.WriteAllText(reportPath, JsonConvert.SerializeObject(report, Formatting.Indented));
        Console.WriteLine($"Wrote report to {reportPath}");
        return 0;
    }

    private static AiModelResult<float, Tensor<float>, Tensor<float>> LoadProgramSynthesisModel(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException("Model file not found.", modelPath);
        }

        return AiModelResult<float, Tensor<float>, Tensor<float>>.LoadModel(
            modelPath,
            metadata =>
            {
                if (!metadata.AdditionalInfo.TryGetValue("ModelName", out var modelNameObj) ||
                    modelNameObj is not string modelName)
                {
                    throw new InvalidOperationException("Model metadata missing 'ModelName'.");
                }

                var language = ParseEnum(metadata.AdditionalInfo.TryGetValue("TargetLanguage", out var langObj) ? langObj?.ToString() : null, ProgramLanguage.Generic);
                var tokenizer = ProgramSynthesisTokenizerFactory.CreateDefault(language, splitIdentifiers: true);
                var modelArch = modelName switch
                {
                    "CodeT5" => CodeModelArch.CodeT5,
                    "CodeBERT" => CodeModelArch.CodeBERT,
                    "GraphCodeBERT" => CodeModelArch.GraphCodeBERT,
                    _ => throw new NotSupportedException($"Unsupported ModelName '{modelName}'.")
                };

                var task = ParseEnum(metadata.AdditionalInfo.TryGetValue("CodeTaskType", out var taskObj) ? taskObj?.ToString() : null, CodeTask.Generation);
                var maxSeq = ParseInt(metadata.AdditionalInfo.TryGetValue("MaxSequenceLength", out var maxSeqObj) ? maxSeqObj?.ToString() : null, 512);
                var vocab = ParseInt(metadata.AdditionalInfo.TryGetValue("VocabularySize", out var vocabObj) ? vocabObj?.ToString() : null, tokenizer.VocabularySize);
                var numEncoder = ParseInt(metadata.AdditionalInfo.TryGetValue("NumEncoderLayers", out var encObj) ? encObj?.ToString() : null, 6);
                var numDecoder = ParseInt(metadata.AdditionalInfo.TryGetValue("NumDecoderLayers", out var decObj) ? decObj?.ToString() : null, modelArch == CodeModelArch.CodeT5 ? 6 : 0);
                var numHeads = ParseInt(metadata.AdditionalInfo.TryGetValue("NumHeads", out var headsObj) ? headsObj?.ToString() : null, 8);
                var modelDim = ParseInt(metadata.AdditionalInfo.TryGetValue("ModelDimension", out var dimObj) ? dimObj?.ToString() : null, 512);
                var ffDim = ParseInt(metadata.AdditionalInfo.TryGetValue("FeedForwardDimension", out var ffObj) ? ffObj?.ToString() : null, 2048);
                var dropout = ParseDouble(metadata.AdditionalInfo.TryGetValue("DropoutRate", out var drObj) ? drObj?.ToString() : null, 0.1);
                var usePos = ParseBool(metadata.AdditionalInfo.TryGetValue("UsePositionalEncoding", out var peObj) ? peObj?.ToString() : null, true);

                var architecture = new CodeSynthesisArchitecture<float>(
                    synthesisType: SynthesisType.Neural,
                    targetLanguage: language,
                    codeTaskType: task,
                    numEncoderLayers: numEncoder,
                    numDecoderLayers: numDecoder,
                    numHeads: numHeads,
                    modelDimension: modelDim,
                    feedForwardDimension: ffDim,
                    maxSequenceLength: maxSeq,
                    vocabularySize: vocab,
                    maxProgramLength: maxSeq,
                    dropoutRate: dropout,
                    usePositionalEncoding: usePos,
                    useDataFlow: modelArch == CodeModelArch.GraphCodeBERT,
                    complexity: NetworkComplexity.Medium,
                    inputSize: modelDim,
                    outputSize: vocab,
                    layers: null);

                return CreateModel<float>(modelArch, architecture, tokenizer);
            });
    }

    private static CodeSynthesisArchitecture<T> CreateDefaultArchitecture<T>(
        CodeModelArch arch,
        ProgramLanguage language,
        CodeTask task,
        ITokenizer tokenizer)
    {
        var vocabSize = Math.Max(1024, tokenizer.VocabularySize);
        return new CodeSynthesisArchitecture<T>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: language,
            codeTaskType: task,
            numEncoderLayers: 6,
            numDecoderLayers: arch == CodeModelArch.CodeT5 ? 6 : 0,
            numHeads: 8,
            modelDimension: 512,
            feedForwardDimension: 2048,
            maxSequenceLength: 512,
            vocabularySize: vocabSize,
            maxProgramLength: 512,
            dropoutRate: 0.1,
            usePositionalEncoding: true,
            useDataFlow: arch == CodeModelArch.GraphCodeBERT,
            complexity: NetworkComplexity.Medium,
            inputSize: 512,
            outputSize: vocabSize,
            layers: null);
    }

    private static CodeModelBase<T> CreateModel<T>(CodeModelArch arch, CodeSynthesisArchitecture<T> architecture, ITokenizer tokenizer)
    {
        return arch switch
        {
            CodeModelArch.CodeT5 => new CodeT5<T>(architecture, lossFunction: new CrossEntropyLoss<T>(), optimizer: null, tokenizer: tokenizer),
            CodeModelArch.CodeBERT => new CodeBERT<T>(architecture, lossFunction: new CrossEntropyLoss<T>(), optimizer: null, tokenizer: tokenizer),
            CodeModelArch.GraphCodeBERT => new GraphCodeBERT<T>(architecture, lossFunction: new CrossEntropyLoss<T>(), optimizer: null, tokenizer: tokenizer),
            _ => throw new ArgumentOutOfRangeException(nameof(arch), arch, "Unknown architecture.")
        };
    }

    private static Tensor<T> EncodeToTokenIdTensor<T>(ITokenizer tokenizer, string text, int maxLength)
    {
        var options = new AiDotNet.Tokenization.Models.EncodingOptions
        {
            AddSpecialTokens = true,
            Truncation = true,
            MaxLength = maxLength
        };

        var ids = tokenizer.Encode(text ?? string.Empty, options).TokenIds;
        var seq = Math.Max(1, Math.Min(maxLength, ids.Count));
        var tensor = new Tensor<T>([seq, 1, 1]);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < seq; i++)
        {
            tensor[i, 0, 0] = numOps.FromDouble(ids[i]);
        }
        return tensor;
    }

    private static List<TrainingSample> ReadTrainingSamples(string path, int? maxSamples)
    {
        var samples = new List<TrainingSample>();
        using var reader = new StreamReader(File.OpenRead(path));

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var sample = TryParseTrainingSample(line);
            if (sample is null)
            {
                continue;
            }

            samples.Add(sample);
            if (maxSamples.HasValue && samples.Count >= maxSamples.Value)
            {
                break;
            }
        }

        return samples;
    }

    private static TrainingSample? TryParseTrainingSample(string jsonLine)
    {
        try
        {
            var obj = JObject.Parse(jsonLine);
            var prompt = (string?)obj["prompt"] ?? (string?)obj["source"] ?? (string?)obj["input"] ?? string.Empty;
            var target = (string?)obj["completion"] ?? (string?)obj["target"] ?? (string?)obj["output"] ?? string.Empty;
            if (string.IsNullOrWhiteSpace(prompt) || string.IsNullOrWhiteSpace(target))
            {
                return null;
            }

            return new TrainingSample(prompt, target);
        }
        catch
        {
            return null;
        }
    }

    private static void ShuffleInPlace<TItem>(IList<TItem> items, Random random)
    {
        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    private static string? GetArgValue(string[] args, string name)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (!string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (i + 1 >= args.Length)
            {
                return null;
            }

            return args[i + 1];
        }

        return null;
    }

    private static int ParseInt(string? value, int defaultValue)
        => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) ? parsed : defaultValue;

    private static int? ParseNullableInt(string? value)
        => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) ? parsed : null;

    private static double ParseDouble(string? value, double defaultValue)
        => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : defaultValue;

    private static bool ParseBool(string? value, bool defaultValue)
        => bool.TryParse(value, out var parsed) ? parsed : defaultValue;

    private static TEnum ParseEnum<TEnum>(string? value, TEnum defaultValue)
        where TEnum : struct
        => !string.IsNullOrWhiteSpace(value) && Enum.TryParse<TEnum>(value, ignoreCase: true, out var parsed) ? parsed : defaultValue;

    private sealed record TrainingSample(string Prompt, string Target);

    private enum CodeModelArch
    {
        CodeT5,
        CodeBERT,
        GraphCodeBERT
    }
}
