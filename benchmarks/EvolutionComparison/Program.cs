using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Prompts;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Benchmarks.EvolutionComparison;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Length != 7 || !int.TryParse(args[3], out int iterations) || iterations is < 1 or > 64 ||
            !uint.TryParse(args[4], out uint seed) || string.IsNullOrWhiteSpace(args[2]) || args[6] is not ("controlled" or "native-bounded"))
        {
            Console.Error.WriteLine("Usage: <initial.py> <new-output.json> <model> <iterations 1..64> <uint-seed> <task-description.txt> <controlled|native-bounded>");
            return 2;
        }
        string source = ReadBounded(args[0], 65536);
        string description = ReadBounded(args[5], 16384);
        using var broker = new BrokerClient(args[2]);
        using var output = new FileStream(args[1], FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        var programOptions = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.Python,
            TaskDescription = description,
            MaxProgramChars = 65536
        };
        var variationOptions = new LlmProgramVariationOptions
        {
            Mode = ProgramEvolutionMode.FullRewrite,
            MaxProposalRetries = 0,
            MaxRecordedAttempts = 64
        };
        ProgramPromptBuilder? promptBuilder = null;
        if (args[6] == "controlled")
        {
            promptBuilder = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions
            {
                EvolutionMode = ProgramPromptEvolutionMode.FullRewrite,
                SystemMessageMode = ProgramPromptSystemMessageMode.Literal,
                SystemMessage = "Optimize the supplied Python program for this task. Return the complete program in a python code fence. " +
                    "Preserve its interface and correctness. Do not access tools or evaluate it yourself.\nTask:\n" + description,
                UseTemplateStochasticity = false,
                MaxProgramSnippetChars = 65536,
                MaxPromptChars = 131072,
                TemplateOverrides = new Dictionary<ProgramPromptTemplateKey, string>
                {
                    [ProgramPromptTemplateKey.FullRewriteUser] = "Parent program:\n```python\n{current_program}\n```"
                }
            }, programOptions);
        }
        var variation = new LlmProgramVariationOperator<double>(broker, programOptions, variationOptions, promptBuilder: promptBuilder);
        var evaluator = new DelegateProgramFitnessEvaluator(async (genome, _, cancellation) =>
        {
            JsonElement receipt = await broker.CallAsync("evaluate", new { code = genome.Source }, cancellation);
            if (receipt.GetProperty("candidate_hash").GetString() != Hash(Encoding.UTF8.GetBytes(genome.Source)) ||
                receipt.GetProperty("unknown_work").GetBoolean())
                throw new InvalidDataException("Shared evaluator did not reconcile the dispatched source.");
            string? status = receipt.GetProperty("status").GetString();
            if (status is not ("valid" or "invalid")) throw new InvalidDataException("Unknown evaluator status.");
            double work = receipt.GetProperty("work_units").GetDouble();
            double quality = status == "valid" ? receipt.GetProperty("quality").GetDouble() : -1e300;
            if (!double.IsFinite(work) || work < 0 || !double.IsFinite(quality)) throw new InvalidDataException("Nonfinite measurement.");
            return new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, quality,
                constraintViolations: new[] { status == "valid" ? 0d : 1d }, costUnits: work);
        }, versionHash: "shared-program-broker-v1");
        var task = new ProgramEvolutionTask(evaluator,
            new ProgramDescriptorSet(new[] { new ProgramLengthDescriptor() }), programOptions);
        var options = new EvolutionEngineOptions
        {
            RunId = "us02-program-comparison",
            Seed = seed,
            MaxEvaluationAttempts = iterations + 1,
            MaxProposals = iterations + 1,
            MaxGenerations = iterations + 1,
            ProposalBatchSize = 1,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            InspirationCount = 3,
            MigrationInterval = 0,
            EnableEvaluationCache = false,
            EvaluationGracePeriod = null
        };
        try
        {
            var observer = new EvaluationRecorder();
            var engine = new EvolutionEngine<ProgramGenome>(task, variation,
                _ => new MapElitesArchive<ProgramGenome>(new[] { new EvolutionDescriptorDefinition("length", 0, 65536, 64) }), options,
                observer: observer);
            EvolutionRunResult<ProgramGenome> result = await engine.RunAsync(new[] { new ProgramGenome(source, ProgramLanguage.Python) });
            await JsonSerializer.SerializeAsync(output, new
            {
                schema = "aidotnet-program-broker-run-v1",
                mode = args[6],
                initial_program_hash = Hash(Encoding.UTF8.GetBytes(source)),
                requested_model = args[2],
                iterations,
                seed,
                engine = new { result.StopReason, result.Counters, result.StateHash },
                usage = variation.GetUsage(),
                attempts = variation.GetRecentAttempts(),
                evaluations = observer.Evaluations,
                artifacts = new[] { typeof(Program).Assembly.Location, typeof(ProgramGenome).Assembly.Location,
                    typeof(EvolutionEngineOptions).Assembly.Location }.ToDictionary(path => Path.GetFileName(path)!, path => Hash(File.ReadAllBytes(path))),
                limitations = "Native AiDotNet prompts with full rewrite/no retries; sequential one-island length archive. Broker receipts determine actual work. No candidate execution occurs in this host. Provider token usage remains in broker transport receipts."
            });
            return 0;
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            await JsonSerializer.SerializeAsync(output, new { schema = "aidotnet-program-broker-run-v1", status = "failed", error = error.GetType().Name });
            return 1;
        }
    }

    private static string ReadBounded(string path, int maximum)
    {
        using var stream = File.OpenRead(path);
        if (stream.Length > maximum) throw new InvalidDataException("Benchmark input exceeds its byte bound.");
        using var reader = new StreamReader(stream, new UTF8Encoding(false, true));
        return reader.ReadToEnd();
    }

    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class EvaluationRecorder : IEvolutionObserver<ProgramGenome>
    {
        internal List<EvolutionEvaluation> Evaluations { get; } = new();
        public ValueTask OnEventAsync(EvolutionEvent<ProgramGenome> evolutionEvent, CancellationToken cancellationToken = default)
        {
            if (evolutionEvent.Kind == EvolutionEventKind.Evaluated && evolutionEvent.Evaluation is { } evaluation)
                Evaluations.Add(evaluation);
            return default;
        }
    }
}

internal sealed class BrokerClient : IChatClient<double>, IDisposable
{
    private readonly HttpClient _http;
    public BrokerClient(string model)
    {
        ModelId = model;
        var address = new Uri(Environment.GetEnvironmentVariable("EVOLUTION_BROKER_ENDPOINT") ?? "", UriKind.Absolute);
        if (address.Scheme != "http" || address.Host != "127.0.0.1" || address.UserInfo.Length != 0 ||
            address.AbsolutePath != "/" || address.Query.Length != 0 || address.Fragment.Length != 0)
            throw new InvalidDataException("Only the local benchmark broker is permitted.");
        string capability = Environment.GetEnvironmentVariable("EVOLUTION_BROKER_CAPABILITY") ?? "";
        if (capability.Length != 64 || capability.Any(value => !Uri.IsHexDigit(value))) throw new InvalidDataException("Missing broker capability.");
        _http = new HttpClient(new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false })
        {
            BaseAddress = address,
            Timeout = TimeSpan.FromSeconds(310),
            MaxResponseContentBufferSize = 256 * 1024
        };
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", capability);
    }

    public string ModelId { get; }

    public async Task<JsonElement> CallAsync(string operation, object payload, CancellationToken cancellation)
    {
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(payload);
        if (bytes.Length > 256 * 1024) throw new InvalidDataException("Broker request exceeds its byte bound.");
        // JsonContent streams with chunked framing; the bounded broker requires
        // a verified Content-Length before reading any payload.
        using var content = new ByteArrayContent(bytes);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        using var response = await _http.PostAsync(operation, content, cancellation);
        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(await response.Content.ReadAsByteArrayAsync(cancellation));
        if (document.RootElement.GetProperty("status").GetString() != "ok") throw new InvalidDataException("Broker rejected work.");
        return document.RootElement.GetProperty("result").Clone();
    }

    public async Task<ChatResponse> GetResponseAsync(IReadOnlyList<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var payload = new
        {
            system = string.Join("\n", messages.Where(message => message.Role == ChatRole.System).Select(message => message.Text)),
            messages = messages.Where(message => message.Role != ChatRole.System)
                .Select(message => new { role = message.Role.ToString().ToLowerInvariant(), content = message.Text }).ToArray()
        };
        JsonElement result = await CallAsync("model", payload, cancellationToken);
        return new ChatResponse(ChatMessage.Assistant(result.GetString() ?? throw new InvalidDataException("Missing model text.")), modelId: ModelId);
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null, CancellationToken cancellationToken = default) => throw new NotSupportedException("Benchmark generation is non-streaming.");

    public void Dispose() => _http.Dispose();
}
