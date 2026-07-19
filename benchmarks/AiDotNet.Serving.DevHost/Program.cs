using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serving.Controllers;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.AspNetCore.Mvc.ApplicationParts;
using Microsoft.AspNetCore.Mvc.Controllers;

static int EnvInt(string name, int fallback) =>
    int.TryParse(Environment.GetEnvironmentVariable(name), out var v) && v > 0 ? v : fallback;

// Engine selection: DEVHOST_GPU=1 puts every model forward on the auto-detected GPU (OpenCL here);
// default is the CPU engine. Set process-wide before any model runs.
if (Environment.GetEnvironmentVariable("DEVHOST_GPU") == "1")
{
    bool ok = AiDotNetEngine.AutoDetectAndConfigureGpu();
    Console.WriteLine(ok && AiDotNetEngine.Current is DirectGpuTensorEngine
        ? $"[DevHost] GPU engine active: {AiDotNetEngine.Current.GetType().Name}"
        : "[DevHost] GPU requested but unavailable — falling back to CPU engine.");
}
else
{
    Console.WriteLine("[DevHost] CPU engine.");
}

// ---------------------------------------------------------------------------
// DEV/BENCHMARK host. Wires the REAL OpenAI controller + generation engine +
// tokenizer registry, with NO auth/DB, plus a SYNTHETIC generative model, so
// `adnbench --backend openai` can drive /v1 end-to-end and measure the serving
// layer. The forward pass is synthetic: numbers reflect engine overhead, not
// real model compute.
// ---------------------------------------------------------------------------

const string ModelName = "dev-lm";
int port = int.TryParse(Environment.GetEnvironmentVariable("DEVHOST_PORT"), out var p) ? p : 5090;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(o => o.ListenLocalhost(port));

// Only expose OpenAiController (the other AiDotNet.Serving controllers need auth/DB/Stripe deps).
builder.Services
    .AddControllers()
    .AddApplicationPart(typeof(OpenAiController).Assembly)
    .AddNewtonsoftJson()
    .ConfigureApplicationPartManager(mgr =>
    {
        foreach (var existing in mgr.FeatureProviders.OfType<ControllerFeatureProvider>().ToList())
            mgr.FeatureProviders.Remove(existing);
        mgr.FeatureProviders.Add(new OnlyOpenAiController());
    });

// The exact services OpenAiController depends on — real implementations.
builder.Services.AddSingleton<IModelRepository, ModelRepository>();
builder.Services.AddSingleton<ITokenizerRegistry, TokenizerRegistry>();
builder.Services.AddSingleton<ITextGenerationService, TextGenerationService>();

var app = builder.Build();

// Register a synthetic generative model + an ASCII character tokenizer.
var repo = app.Services.GetRequiredService<IModelRepository>();
var tokenizers = app.Services.GetRequiredService<ITokenizerRegistry>();

// DEVHOST_GGUF=<path> serves a REAL llama.cpp GGUF checkpoint: its decoder is rebuilt + weight-loaded
// through the pretrained facade and its byte-level BPE tokenizer is read from the same file, so the server
// runs the model at parity with llama.cpp (identical weights + identical tokenization). Everything below
// (paged continuous batching, prefill/decode kernels) is unchanged.
string? ggufPath = Environment.GetEnvironmentVariable("DEVHOST_GGUF");
if (!string.IsNullOrWhiteSpace(ggufPath))
{
    if (!File.Exists(ggufPath))
    {
        Console.Error.WriteLine($"[DevHost] GGUF file not found: {ggufPath}");
        return;
    }

    Console.WriteLine($"[DevHost] loading GGUF checkpoint: {ggufPath}");
    var (ggufModel, ggufTokenizer) = PretrainedLoader<float>.LoadGgufWithTokenizer(ggufPath);
    var ggufWrapper = new ServableModelWrapper<float>(
        ModelName, ggufModel, inputShape: new[] { 1 },
        enableSpeculativeDecoding: false, generationForward: ggufModel.Predict);
    Console.WriteLine(
        $"[DevHost] GGUF model={ggufModel.GetType().Name} vocab={ggufTokenizer.VocabularySize} " +
        $"incrementalPaged={ggufWrapper.SupportsIncrementalGeneration} batchedPrefill={ggufWrapper.SupportsBatchedPrefill}");

    repo.LoadModel<float>(ModelName, ggufWrapper);
    tokenizers.Register(ModelName, ggufTokenizer);

    app.MapControllers();
    app.Logger.LogInformation(
        "DevHost ready on http://localhost:{Port} | model={Model} (GGUF)", port, ModelName);
    app.Run();
    return;
}

var tokenizer = CharacterTokenizer.CreateAscii();
int vocab = tokenizer.VocabularySize;

// Synthetic token-level forward: [1, seq] token ids -> [1, seq, vocab] logits.
// Peaks at a position-varying id in the ASCII range (high ids) so decoded output is
// readable-ish and generation runs to max_tokens (BERT default has no EOS token).
Func<Tensor<float>, Tensor<float>> synthForward = input =>
{
    int seq = input.Shape[input.Shape.Length - 1];
    var logits = new Tensor<float>(new[] { 1, seq, vocab });
    for (int pos = 0; pos < seq; pos++)
    {
        int peak = vocab - 1 - ((pos * 7 + 3) % Math.Max(1, vocab - 6));
        if (peak < 0) peak = vocab - 1;
        logits[new[] { 0, pos, peak }] = 30f;
    }
    return logits;
};

// By default, serve a small REAL per-position transformer LM (Embedding -> MHA -> Dense) so requests
// drive the UNIFIED paged continuous-batching engine (batched prefill + decode + speculation, prefix
// sharing). Set DEVHOST_SYNTHETIC=1 to instead use the fast synthetic Func forward, which routes the
// stateless per-request path and measures pure engine/HTTP overhead rather than the batching win.
bool synthetic = Environment.GetEnvironmentVariable("DEVHOST_SYNTHETIC") == "1";
ServableModelWrapper<float> model;
if (synthetic)
{
    model = new ServableModelWrapper<float>(
        ModelName, inputDimension: 1, outputDimension: vocab, predictFunc: v => v, generationForward: synthForward);
}
else
{
    // Size is env-configurable so the benchmark can run a compute-representative model (a tiny embDim is
    // dominated by kernel-launch/HTTP overhead and hides where the GPU actually helps). Each block adds
    // attention + a position-wise FFN (up-project GELU -> down-project) to mirror a real decoder's per-token cost.
    int embDim = EnvInt("DEVHOST_EMBDIM", 64);
    int heads = EnvInt("DEVHOST_HEADS", 4);
    int ffn = EnvInt("DEVHOST_FFN", embDim * 4);
    int blocks = EnvInt("DEVHOST_LAYERS", 1);
    var layers = new List<AiDotNet.Interfaces.ILayer<float>> { new EmbeddingLayer<float>(vocab, embDim) };
    for (int b = 0; b < blocks; b++)
    {
        layers.Add(new MultiHeadAttentionLayer<float>(heads, embDim / heads,
            activationFunction: new IdentityActivation<float>()));
        layers.Add(new DenseLayer<float>(ffn, activationFunction: new GELUActivation<float>()));
        layers.Add(new DenseLayer<float>(embDim, activationFunction: new IdentityActivation<float>()));
    }
    layers.Add(new DenseLayer<float>(vocab, activationFunction: new IdentityActivation<float>()));
    var architecture = new NeuralNetworkArchitecture<float>(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
        complexity: NetworkComplexity.Simple, inputSize: 1, outputSize: vocab, layers: layers);
    var lm = new NeuralNetwork<float>(architecture);
    var pv = lm.GetParameters();
    var det = new float[pv.Length];
    for (int i = 0; i < det.Length; i++) det[i] = ((i % 17) - 8) / 16.0f;
    lm.UpdateParameters(new Vector<float>(det));
    model = new ServableModelWrapper<float>(
        ModelName, lm, inputShape: new[] { 1 }, enableSpeculativeDecoding: false, generationForward: lm.Predict);
    Console.WriteLine($"[DevHost] model embDim={embDim} heads={heads} ffn={ffn} blocks={blocks} vocab={vocab} params={pv.Length}");
}

Console.WriteLine($"[DevHost] incrementalPaged={model.SupportsIncrementalGeneration} batchedPrefill={model.SupportsBatchedPrefill}");
repo.LoadModel<float>(ModelName, model);
tokenizers.Register(ModelName, tokenizer);

app.MapControllers();

app.Logger.LogInformation("DevHost ready on http://localhost:{Port} | model={Model} vocab={Vocab}", port, ModelName, vocab);

app.Run();

/// <summary>Restricts MVC controller discovery to just the OpenAI controller.</summary>
internal sealed class OnlyOpenAiController : ControllerFeatureProvider
{
    protected override bool IsController(TypeInfo typeInfo) => typeInfo == typeof(OpenAiController).GetTypeInfo();
}
