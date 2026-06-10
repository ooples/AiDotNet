# AiDotNet Agentic Orchestration

A cohesive, type-safe .NET subsystem for building agents and multi-agent systems that reaches
feature-parity-plus with Semantic Kernel and LangGraph, and adds four capabilities neither offers:
**local in-process inference**, **self-improving orchestration**, a **typed deterministic graph**, and a
**unified ML + LLM runtime**.

Everything lives under `AiDotNet.Agentic.*` and is built on one numeric type parameter `T` (e.g. `float`,
`double`) shared end-to-end, so an agent and the model it drives agree on one tensor element type.

> **For Beginners:** an *agent* is a program that uses a language model to decide what to do — answer, call
> a tool, or hand off to another agent. This subsystem gives you the model clients, the tools, the control
> flow (graphs/teams), memory, observability, and even a model that runs entirely on your machine — all with
> the same code.

---

## Layers

| Layer | Namespace | What it gives you |
|---|---|---|
| Model | `AiDotNet.Agentic.Models` | `IChatClient<T>` — message-based, streaming, native tool-calling, structured output |
| Connectors | `AiDotNet.Agentic.Models.Connectors` | OpenAI, Anthropic, Azure OpenAI, **Ollama**, **Mistral**, **Gemini**, **Cohere**, MEAI adapter |
| Local engine | `AiDotNet.Agentic.Models.Local` | `LocalEngineChatClient<T>` — in-process generation over AiDotNet's own networks |
| Tools | `AiDotNet.Agentic.Tools` | `IAgentTool`, `ToolCollection`, source-generated JSON schema, structured output |
| Graph | `AiDotNet.Agentic.Graph` | typed `StateGraph<TState>`, checkpointing, HITL, subgraphs, fan-out, reward-gated edges |
| Agents | `AiDotNet.Agentic.Agents` | `AgentExecutor<T>`, `SupervisorAgent<T>`, `Swarm<T>` |
| Memory | `AiDotNet.Agentic.Memory` | conversation threads + long-term semantic memory |
| Self-improving | `AiDotNet.Agentic.SelfImproving` | trajectory capture, eval, learned routing, prompt-opt, LoRA data |
| Pipeline | `AiDotNet.Agentic.Pipeline` | filters/middleware, guardrails, OpenTelemetry |
| MCP | `AiDotNet.Agentic.Mcp` | Model Context Protocol client **and** server |

---

## Quick start

### A single tool-calling agent

```csharp
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Agentic.Tools;

IChatClient<double> model = new OpenAIChatClient<double>(apiKey, "gpt-4o");

var tools = new ToolCollection()
    .AddDelegate("add", "Adds two integers.", (int a, int b) => a + b);

var agent = new AgentExecutor<double>(model, tools, new AgentExecutorOptions
{
    SystemPrompt = "You are a careful assistant.",
});

AgentRunResult result = await agent.RunAsync("What is 2 + 3?");
Console.WriteLine(result.FinalText);
```

The executor runs a **native** tool-calling loop: it calls the model with the tools, runs any tool the model
requests, feeds the results back, and repeats until a final answer.

### A supervisor team

```csharp
var researcher = new AgentExecutor<double>(model, researchTools,
    new AgentExecutorOptions { Name = "researcher", Description = "Finds facts." });
var writer = new AgentExecutor<double>(model, tools: null,
    new AgentExecutorOptions { Name = "writer", Description = "Writes prose." });

var supervisor = new SupervisorAgent<double>(model, new IAgent<double>[] { researcher, writer });
var answer = await supervisor.RunAsync("Research X and write a summary.");
```

A `Swarm<double>` is the peer-to-peer alternative: members hand off control to each other over one shared
conversation.

### Conversation + long-term memory

```csharp
var threaded = new ThreadedAgent<double>(agent, new InMemoryConversationStore());
await threaded.RunAsync("conv-1", "My name is Alice.");
await threaded.RunAsync("conv-1", "What's my name?"); // remembers Alice

var memory = new EmbeddingAgentMemoryStore<double>(embeddingModel);
await memory.AddAsync("The user prefers metric units.");
var recalling = new MemoryAugmentedAgent<double>(agent, memory);
```

---

## The four differentiators

### 1. Local in-process inference

`LocalEngineChatClient<T>` runs generation **inside your process** — no network, no API key — over AiDotNet's
own networks (e.g. `MambaLanguageModel`, `GLALanguageModel`) via `NeuralNetworkCausalLanguageModel<T>`.

```csharp
var lm = new NeuralNetworkCausalLanguageModel<double>(myTransformer, vocabSize);
IChatClient<double> local = new LocalEngineChatClient<double>(lm, tokenizer, options: new LocalEngineOptions
{
    Sampling = new LocalSamplingOptions { Temperature = 0.7, TopP = 0.95 },
});
var agent = new AgentExecutor<double>(local); // same agent code, no cloud
```

It supports greedy/temperature/top-k/top-p sampling, **beam search**, a **KV-cache fast path**
(`IIncrementalCausalLanguageModel<T>`), and **constrained decoding** (`ITokenConstraint`) for guaranteed
structured output the engine enforces at the logits — something cloud APIs can only approximate.

### 2. Self-improving orchestration

Capture every run, evaluate it, and learn:

```csharp
var store = new InMemoryTrajectoryStore();
var traced = new TracingAgent<double>(agent, store);          // records each run
// ... run many tasks ...
var report = await new TrajectoryEvaluationRunner(evaluator).EvaluateStoreAsync(store);

var router = new LearnedAgentRouter<double>(new[] { fast, accurate });
router.LearnFrom(await store.GetAllAsync());                  // routes to the best agent per context

var dataset = new RewardFilteredDatasetBuilder(minReward: 0.7).Build(await store.GetAllAsync());
// hand `dataset` to the LoRA trainer to fine-tune the local model on its own best work
```

Also includes `PromptOptimizer<T>` (DSPy-like prompt search against a labeled eval set).

### 3. Typed deterministic graph

`StateGraph<TState>` is strongly typed (nodes transform `TState`; no untyped state dict), with conditional
edges and cycles, durable checkpointing (in-memory / JSON-file / SQLite), resume, time-travel replay,
human-in-the-loop interrupts, dynamic fan-out (map-reduce), subgraphs, and reward-gated edges.

### 4. Unified ML + LLM runtime

Any trained `AiModelResult`, regressor/classifier, or RAG pipeline can be wrapped as an `IAgentTool` and
called by an agent — and the same numeric `T` flows from the tensor engine through the model to the agent.

---

## Cross-cutting: filters, guardrails, observability

```csharp
IChatClient<double> hardened = new MiddlewareChatClient<double>(model, new IChatMiddleware[]
{
    new TelemetryChatMiddleware(),                                   // OpenTelemetry GenAI spans + metrics
    new ContentSafetyMiddleware(new DenyListContentModerator(bannedTerms)), // guardrails
});
```

Tool calls have their own pipeline — wrap a sensitive tool in a `MiddlewareAgentTool` with an
`ApprovalToolMiddleware` for human-in-the-loop authorization.

Telemetry is emitted on the `AiDotNet.Agentic` OpenTelemetry source/meter; point any OTel collector at it.

---

## MCP (Model Context Protocol)

**Consume** external MCP servers' tools:

```csharp
var mcp = new McpClient(transport);          // stdio / HTTP / in-memory transport
var mcpTools = await mcp.GetToolsAsync();     // a ToolCollection of remote tools
var agent = new AgentExecutor<double>(model, mcpTools);
```

**Expose** your AiDotNet tools to other MCP clients:

```csharp
var server = new McpServer(myTools);
// host it over stdio/HTTP, or in-process via new InMemoryMcpTransport(server)
```

---

## Delivered follow-ups

- **Hosting/DI** — `AddAgentChatClient` / `AddAgentTools` / `AddAgentExecutor` in `AiDotNet.Serving`.
- **Real MCP transports** — `HttpMcpTransport` (JSON-RPC over HTTP) and `StreamMcpTransport`
  (newline-delimited JSON-RPC over stdio streams), plus the in-memory transport.
- **Weight loaders** — `SafetensorsReader` and `GgufReader` (header, metadata, tensor directory; F32/F16
  **and Q4_0/Q4_1/Q8_0** value reads) + `WeightImporter` (named tensors → a model's flat parameter vector).
- **LLM-as-judge evaluator** (`ChatClientTrajectoryEvaluator`) and a **REINFORCE policy router**
  (`SoftmaxPolicyRouter`).
- **Deterministic record/replay** of model IO (`RecordingChatClient` / `ReplayingChatClient`).
- **Multi-agent-on-graph** — `AddAgentNode` runs any `IAgent<T>` (executor/supervisor/swarm) as a typed
  `StateGraph` node (graph runtime converged onto this branch).
- **Postgres + Redis graph checkpointers** — `PostgresGraphCheckpointer` / `RedisGraphCheckpointer` (opt-in
  packages; integration-tested, skip without a live server).
- **LoRA fine-tuning bridge** — `FineTuningDataConverter` + `LoRAFineTuner.FineTuneFromDatasetAsync` feed a
  reward-filtered dataset into `SupervisedFineTuning`.
- **Benchmark harness** — `AgenticBenchmarks` (agent loop + middleware overhead) + documented head-to-head
  methodology.
- **In-model K/V cache (Mamba)** — `MambaLanguageModel.Step` / `CreateStepState` add a per-token decoding
  path that carries the causal-conv window + selective-scan hidden state, and `MambaCausalLanguageModel`
  exposes it as the `IIncrementalCausalLanguageModel<T>` fast path. Verified mathematically equivalent to the
  full-sequence forward (Gu & Dao 2023) by step-vs-parallel-scan tests at the scan, block, model, and adapter
  levels.

## Remaining work (model-layer / external infrastructure)

Tracked with exact integration points; deliberately not stubbed because they cannot be built **and verified**
within this subsystem:

- **In-model K/V cache for other architectures** — done for Mamba (see *Delivered*). The same per-token
  `Step` treatment can be extended to the other recurrent/attention LMs (GLA, RWKV, Transformer attention
  cache); each needs its own state object + an equivalence test against its full-sequence forward.
- **LoRA fine-tuning execution on a tensor model** — the bridge to `SupervisedFineTuning` is done; running it
  end-to-end needs a text-consuming model (tokenization in front of a tensor model) and real GPU training.
- **Quantized k-quants (Q4_K/Q6_K)** super-block decoding (classic Q4_0/Q4_1/Q8_0 are done) and a built-in
  **per-architecture tensor-name map** (`WeightImporter` takes the caller's name ordering today, since
  networks expose a flat parameter vector rather than named layers).
- **Cross-framework benchmark execution** vs Semantic Kernel & LangGraph — needs those frameworks installed
  in a dedicated environment (the AiDotNet-side harness + methodology are in place).
- **Head-to-head benchmarks vs Semantic Kernel & LangGraph** — a cross-framework measurement harness (those
  frameworks are external to this repo).
