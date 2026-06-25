---
title: "Agentic"
description: "All 149 public types in the AiDotNet.agentic namespace, organized by kind."
section: "API Reference"
---

**149** public types in this namespace, organized by kind.

## Models & Types (89)

| Type | Summary |
|:-----|:--------|
| [`AgentExecutor<T>`](/docs/reference/wiki/agentic/agentexecutor/) | A single-model agent that drives an `IChatClient` in a native tool-calling loop: it calls the model, runs any tools the model requests, feeds the results back, and repeats until the model returns a final answer (or the iteration cap is hit)… |
| [`AgentMemory`](/docs/reference/wiki/agentic/agentmemory/) | A single long-term memory: a piece of text the agent should be able to recall later, with a stable id and optional metadata. |
| [`AgentRunResult`](/docs/reference/wiki/agentic/agentrunresult/) | The outcome of an agent run: the final text answer, the full conversation transcript it produced, how many model calls it took, whether it finished cleanly, and the aggregate token usage. |
| [`AgentTrajectory`](/docs/reference/wiki/agentic/agenttrajectory/) | A captured record of one agent run — the structured "trajectory" the self-improving layer learns from. |
| [`AiDotNetMeaiChatClient<T>`](/docs/reference/wiki/agentic/aidotnetmeaichatclient/) | Adapts an AiDotNet `IChatClient` to a `IChatClient`, so AiDotNet models (including the in-process `LocalEngineChatClient` and the first-party connectors) can be consumed by any code written against the .NET ecosystem's standard chat abstrac… |
| [`AiToolDefinition`](/docs/reference/wiki/agentic/aitooldefinition/) | Describes a tool/function the model is allowed to call: its name, a description, and a JSON-schema describing its parameters. |
| [`AllowedTokenSetConstraint`](/docs/reference/wiki/agentic/allowedtokensetconstraint/) | A constraint that always restricts generation to a fixed set of token ids, regardless of context — for example, "only emit digit tokens" or "only emit tokens from this closed label vocabulary". |
| [`AnthropicChatClient<T>`](/docs/reference/wiki/agentic/anthropicchatclient/) | An `IChatClient` for Anthropic's Claude Messages API with native tool use, streaming, and multimodal (image) input. |
| [`ApprovalToolMiddleware`](/docs/reference/wiki/agentic/approvaltoolmiddleware/) | An `IToolMiddleware` that gates tool execution behind an approval check — human-in-the-loop or policy-based tool authorization. |
| [`AzureOpenAIChatClient<T>`](/docs/reference/wiki/agentic/azureopenaichatclient/) | An `IChatClient` for Azure OpenAI. |
| [`ChatClientTrajectoryEvaluator<T>`](/docs/reference/wiki/agentic/chatclienttrajectoryevaluator/) | An `ITrajectoryEvaluator` that scores a run with a model acting as judge ("LLM-as-judge"): it shows a judging model the task and the agent's answer (and an optional rubric) and parses a 0–1 score from the reply. |
| [`ChatMessage`](/docs/reference/wiki/agentic/chatmessage/) | A single message in a chat conversation: a `ChatRole` plus one or more content parts. |
| [`ChatRequestContext`](/docs/reference/wiki/agentic/chatrequestcontext/) | The mutable request state flowing through a chat-middleware pipeline. |
| [`ChatResponse`](/docs/reference/wiki/agentic/chatresponse/) | The complete result of a non-streaming chat call. |
| [`ChatResponseUpdate`](/docs/reference/wiki/agentic/chatresponseupdate/) | A single incremental update in a streaming chat response. |
| [`ChatUsage`](/docs/reference/wiki/agentic/chatusage/) | Token accounting for a chat request: how many tokens went in and came out. |
| [`CohereChatClient<T>`](/docs/reference/wiki/agentic/coherechatclient/) | An `IChatClient` for Cohere's Chat API, whose bespoke wire format splits a turn into the latest `message`, a `chat_history` of prior turns, and a `preamble` for system instructions. |
| [`CompiledStateGraph<TState>`](/docs/reference/wiki/agentic/compiledstategraph/) | An executable, validated state graph produced by `Compile`. |
| [`ContentSafetyMiddleware`](/docs/reference/wiki/agentic/contentsafetymiddleware/) | A guardrail `IChatMiddleware` that screens the user input before the model is called and/or the model's response after, using an `IContentModerator`. |
| [`DelegateAgentTool`](/docs/reference/wiki/agentic/delegateagenttool/) | An `IAgentTool` backed by an ordinary C# method (a delegate or a reflected method on an instance). |
| [`DelegateChatMiddleware`](/docs/reference/wiki/agentic/delegatechatmiddleware/) | An `IChatMiddleware` backed by a delegate — the quick way to add a one-off filter (logging, a header injection, a guard) without declaring a class. |
| [`DelegateToolMiddleware`](/docs/reference/wiki/agentic/delegatetoolmiddleware/) | An `IToolMiddleware` backed by a delegate — the quick way to add a one-off tool filter (logging, argument fix-up, a guard) without declaring a class. |
| [`DelegateTrajectoryEvaluator`](/docs/reference/wiki/agentic/delegatetrajectoryevaluator/) | An `ITrajectoryEvaluator` backed by a user-supplied scoring function — the general-purpose hook for custom rewards (exact-match against a labeled answer, regex/JSON validity, cost penalties, or any combination). |
| [`DenyListContentModerator`](/docs/reference/wiki/agentic/denylistcontentmoderator/) | A simple `IContentModerator` that blocks content containing any of a configured set of banned terms. |
| [`EmbeddingAgentMemoryStore<T>`](/docs/reference/wiki/agentic/embeddingagentmemorystore/) | A semantic `IAgentMemoryStore` that ranks memories by meaning, not words. |
| [`EvaluationReport`](/docs/reference/wiki/agentic/evaluationreport/) | Aggregate statistics from evaluating a set of trajectories: how many were scored, the reward distribution, and the fraction that met a pass threshold. |
| [`FineTuningDataset`](/docs/reference/wiki/agentic/finetuningdataset/) | A set of reward-filtered `FineTuningExample`s ready to hand to a LoRA / fine-tuning trainer — the bridge from "the agent did well on these runs" to "make the local model better at them". |
| [`FineTuningExample`](/docs/reference/wiki/agentic/finetuningexample/) | One supervised fine-tuning example distilled from a high-reward trajectory: the prompt the agent saw and the completion it produced that earned a good score. |
| [`FiniteStateTokenConstraint`](/docs/reference/wiki/agentic/finitestatetokenconstraint/) | A constraint defined by a finite-state grammar over token ids: the set of allowed next tokens depends on the most recently generated token (the current state). |
| [`FunctionAgentTool`](/docs/reference/wiki/agentic/functionagenttool/) | An `IAgentTool` whose parameter schema is supplied up front and whose execution is a caller-provided delegate — i.e. |
| [`GeminiChatClient<T>`](/docs/reference/wiki/agentic/geminichatclient/) | An `IChatClient` for Google's Gemini models via the `generateContent` API. |
| [`GgufFile`](/docs/reference/wiki/agentic/gguffile/) | A parsed GGUF file: its version, the metadata key/value store (hyperparameters, tokenizer config, etc.), the tensor directory, and access to F32/F16 tensor values. |
| [`GgufTensorInfo`](/docs/reference/wiki/agentic/gguftensorinfo/) | A tensor directory entry in a GGUF file: its name, dimensions, ggml data type, and byte offset within the tensor-data section. |
| [`GraphCheckpoint<TState>`](/docs/reference/wiki/agentic/graphcheckpoint/) | An immutable snapshot of a graph run at a point in time: which node runs next and the state as of that point. |
| [`GraphRunResult<TState>`](/docs/reference/wiki/agentic/graphrunresult/) | The outcome of a human-in-the-loop graph run: either the run completed, or it paused before an interrupt node awaiting input. |
| [`GraphStepUpdate<TState>`](/docs/reference/wiki/agentic/graphstepupdate/) | A streamed update emitted after a single node finishes executing during a graph run: the node's name and the graph state as it stands after that node. |
| [`HttpMcpTransport`](/docs/reference/wiki/agentic/httpmcptransport/) | An `IMcpTransport` that speaks JSON-RPC 2.0 to a remote MCP server over HTTP POST. |
| [`ImageContent`](/docs/reference/wiki/agentic/imagecontent/) | An image content part within a `ChatMessage`, supplied either as raw bytes or as a URI. |
| [`InMemoryAgentMemoryStore`](/docs/reference/wiki/agentic/inmemoryagentmemorystore/) | A process-local `IAgentMemoryStore` that ranks memories by lexical overlap with the query — the fraction of the query's words that appear in the memory. |
| [`InMemoryChatInteractionStore`](/docs/reference/wiki/agentic/inmemorychatinteractionstore/) | A process-local `IChatInteractionStore` holding recorded chat interactions in memory. |
| [`InMemoryConversationStore`](/docs/reference/wiki/agentic/inmemoryconversationstore/) | A process-local `IConversationStore` that keeps thread histories in memory. |
| [`InMemoryGraphCheckpointer<TState>`](/docs/reference/wiki/agentic/inmemorygraphcheckpointer/) | An in-memory `IGraphCheckpointer` that keeps each thread's checkpoint history in a dictionary. |
| [`InMemoryMcpTransport`](/docs/reference/wiki/agentic/inmemorymcptransport/) | An `IMcpTransport` that forwards requests directly to an in-process `McpServer` — no serialization or network. |
| [`InMemoryTrajectoryStore`](/docs/reference/wiki/agentic/inmemorytrajectorystore/) | A process-local `ITrajectoryStore` that keeps captured runs in memory. |
| [`JsonFileConversationStore`](/docs/reference/wiki/agentic/jsonfileconversationstore/) | An `IConversationStore` that persists thread histories to a single JSON file, so conversations survive process restarts without an external database. |
| [`JsonFileGraphCheckpointer<TState>`](/docs/reference/wiki/agentic/jsonfilegraphcheckpointer/) | A durable `IGraphCheckpointer` that persists all threads' checkpoints to a single JSON file, so runs survive process restarts. |
| [`LearnedAgentRouter<T>`](/docs/reference/wiki/agentic/learnedagentrouter/) | A router that learns, from graded `AgentTrajectory` history, which candidate agent performs best and routes new requests accordingly — a contextual reward-weighted bandit. |
| [`LocalEngineChatClient<T>`](/docs/reference/wiki/agentic/localenginechatclient/) | An `IChatClient` that runs entirely in-process over an `ICausalLanguageModel` — no network, no API key, no external service. |
| [`MambaCausalLanguageModel<T>`](/docs/reference/wiki/agentic/mambacausallanguagemodel/) | A KV-cached `IIncrementalCausalLanguageModel` adapter over `MambaLanguageModel`. |
| [`McpClient`](/docs/reference/wiki/agentic/mcpclient/) | A Model Context Protocol (MCP) client: connects to an MCP server over an `IMcpTransport`, lists the tools it offers, and exposes them as `IAgentTool` instances the agent stack can call — so any MCP server's capabilities become available to… |
| [`McpServer`](/docs/reference/wiki/agentic/mcpserver/) | A Model Context Protocol (MCP) server that exposes a `ToolCollection` of AiDotNet tools to any MCP client (Claude Desktop, other agent frameworks, or AiDotNet's own `McpClient`). |
| [`McpToolDescriptor`](/docs/reference/wiki/agentic/mcptooldescriptor/) | Describes a tool advertised by an MCP server: its name, description, and JSON-Schema input contract — the data needed to surface it to a model as a callable tool. |
| [`MeaiChatClient<T>`](/docs/reference/wiki/agentic/meaichatclient/) | Adapts a `IChatClient` (the .NET ecosystem's standard chat abstraction) to AiDotNet's `IChatClient`, so any Microsoft.Extensions.AI connector (OpenAI, Azure, Ollama, etc.) can drive AiDotNet agents and reasoning. |
| [`MemoryAugmentedAgent<T>`](/docs/reference/wiki/agentic/memoryaugmentedagent/) | Wraps any `IAgent` with long-term memory recall: before each run it searches an `IAgentMemoryStore` for memories relevant to the latest user message and injects them as context, so the agent answers with knowledge gathered across previous c… |
| [`MiddlewareAgentTool`](/docs/reference/wiki/agentic/middlewareagenttool/) | An `IAgentTool` decorator that runs a chain of `IToolMiddleware` around an inner tool's execution. |
| [`MiddlewareChatClient<T>`](/docs/reference/wiki/agentic/middlewarechatclient/) | An `IChatClient` decorator that runs a chain of `IChatMiddleware` around an inner client's calls — the composition root for filters/middleware (logging, guardrails, caching, retry, telemetry). |
| [`MistralChatClient<T>`](/docs/reference/wiki/agentic/mistralchatclient/) | An `IChatClient` for the `` platform, whose chat API is OpenAI-compatible. |
| [`ModerationVerdict`](/docs/reference/wiki/agentic/moderationverdict/) | The verdict from moderating a piece of content: whether it is allowed, and if not, why. |
| [`NeuralNetworkCausalLanguageModel<T>`](/docs/reference/wiki/agentic/neuralnetworkcausallanguagemodel/) | Adapts a trained AiDotNet `NeuralNetworkBase` language model (e.g., `MambaLanguageModel`, `GLALanguageModel`, or a Transformer LM head) to the `ICausalLanguageModel` seam, so `LocalEngineChatClient` can run real, fully in-process generation… |
| [`OllamaChatClient<T>`](/docs/reference/wiki/agentic/ollamachatclient/) | An `IChatClient` for a local `` server, which exposes an OpenAI-compatible chat-completions API. |
| [`OpenAIChatClient<T>`](/docs/reference/wiki/agentic/openaichatclient/) | An `IChatClient` for OpenAI's Chat Completions API with native tool calling, streaming, structured output, and multimodal (image) input. |
| [`PromptEvalCase`](/docs/reference/wiki/agentic/promptevalcase/) | One labeled example in a prompt-optimization eval set: an input to send the agent and the expected answer to score its response against. |
| [`PromptOptimizationResult`](/docs/reference/wiki/agentic/promptoptimizationresult/) | The outcome of prompt optimization: the best-scoring prompt plus the full ranked list of candidates. |
| [`PromptOptimizer<T>`](/docs/reference/wiki/agentic/promptoptimizer/) | Selects the best system prompt for an agent by measuring candidate prompts against a labeled eval set — a DSPy-like, evaluation-driven prompt search. |
| [`RecordingChatClient<T>`](/docs/reference/wiki/agentic/recordingchatclient/) | An `IChatClient` decorator that calls a real inner client and records each request/response into an `IChatInteractionStore`. |
| [`ReplayingChatClient<T>`](/docs/reference/wiki/agentic/replayingchatclient/) | An `IChatClient` that serves responses from a recorded `IChatInteractionStore` — deterministic replay without calling any model. |
| [`SafetensorsFile`](/docs/reference/wiki/agentic/safetensorsfile/) | A parsed safetensors file: the tensor table of contents plus access to each tensor's raw bytes and a conversion to `Double` values. |
| [`SafetensorsTensor`](/docs/reference/wiki/agentic/safetensorstensor/) | Metadata for one tensor inside a safetensors file: its name, dtype, shape, and byte range within the data section. |
| [`ScoredMemory`](/docs/reference/wiki/agentic/scoredmemory/) | A memory paired with its relevance score for a particular query, as returned by `CancellationToken)`. |
| [`ScoredPrompt`](/docs/reference/wiki/agentic/scoredprompt/) | A candidate prompt paired with its mean score over the eval set. |
| [`SoftmaxPolicyRouter<T>`](/docs/reference/wiki/agentic/softmaxpolicyrouter/) | A routing policy over candidate agents, trained by REINFORCE: it keeps a learnable preference weight per agent (optionally per context), selects via a softmax over those weights, and updates them with a policy-gradient step from observed re… |
| [`StateGraph<TState>`](/docs/reference/wiki/agentic/stategraph/) | A builder for a typed state graph: register nodes (state transformers), wire them with fixed or conditional edges (cycles allowed), set an entry point, then `Compile` into an executable `CompiledStateGraph`. |
| [`StreamMcpTransport`](/docs/reference/wiki/agentic/streammcptransport/) | An `IMcpTransport` that speaks newline-delimited JSON-RPC 2.0 over a reader/writer pair — the framing MCP uses over stdio. |
| [`StreamingToolCallUpdate`](/docs/reference/wiki/agentic/streamingtoolcallupdate/) | An incremental fragment of a tool call arriving over a streaming response. |
| [`SupervisorAgent<T>`](/docs/reference/wiki/agentic/supervisoragent/) | A coordinator agent that supervises a team of specialized worker `IAgent` instances and routes work to them via native tool-calling. |
| [`SwarmMember<T>`](/docs/reference/wiki/agentic/swarmmember/) | One participant in a `Swarm`: a named persona with its own chat model, instructions, tools, and the set of peers it is allowed to hand control to. |
| [`Swarm<T>`](/docs/reference/wiki/agentic/swarm/) | A peer-to-peer multi-agent team where control transfers between members over one shared conversation. |
| [`TelemetryChatMiddleware`](/docs/reference/wiki/agentic/telemetrychatmiddleware/) | An `IChatMiddleware` that emits OpenTelemetry GenAI telemetry for each chat call: a client span tagged with the operation, response model, finish reason, and token usage, plus operation-count and token-usage metrics. |
| [`TextContent`](/docs/reference/wiki/agentic/textcontent/) | A plain-text content part within a `ChatMessage`. |
| [`ThreadedAgent<T>`](/docs/reference/wiki/agentic/threadedagent/) | Wraps any `IAgent` with conversation memory: each run is tied to a thread id, so prior turns are loaded from an `IConversationStore` and prepended to the new input, and the new user/assistant turn is persisted afterwards. |
| [`TokenSampler<T>`](/docs/reference/wiki/agentic/tokensampler/) | Selects the next token id from a logits vector according to `LocalSamplingOptions`: greedy (argmax) when temperature is zero, otherwise temperature-scaled softmax sampling restricted by optional top-k and top-p filters. |
| [`TokenizerGenerationAdapter`](/docs/reference/wiki/agentic/tokenizergenerationadapter/) | Bridges a full repo `ITokenizer` to the engine's minimal `IGenerationTokenizer` seam, so any AiDotNet tokenizer can drive `LocalEngineChatClient`. |
| [`ToolCallContent`](/docs/reference/wiki/agentic/toolcallcontent/) | A request, emitted by the assistant, to invoke a named tool/function with JSON arguments. |
| [`ToolCollection`](/docs/reference/wiki/agentic/toolcollection/) | A named set of executable tools: registers `IAgentTool` instances, exposes their `AiToolDefinition`s for a chat request, and dispatches model tool-calls to the right tool. |
| [`ToolInvocationContext`](/docs/reference/wiki/agentic/toolinvocationcontext/) | The mutable state flowing through a tool-invocation middleware pipeline: which tool is being called, the (rewritable) arguments, and a shared property bag. |
| [`ToolInvocationResult`](/docs/reference/wiki/agentic/toolinvocationresult/) | The outcome of executing an `IAgentTool`: the text fed back to the model plus a flag indicating whether the tool failed. |
| [`ToolResultContent`](/docs/reference/wiki/agentic/toolresultcontent/) | The result of executing a tool, fed back to the model to continue a tool-calling turn. |
| [`TracingAgent<T>`](/docs/reference/wiki/agentic/tracingagent/) | Wraps any `IAgent` and records each run as an `AgentTrajectory` in a `ITrajectoryStore`, without altering the agent's behavior. |
| [`TrajectoryEvaluationRunner`](/docs/reference/wiki/agentic/trajectoryevaluationrunner/) | Runs an `ITrajectoryEvaluator` over trajectories, annotating each with its `Reward` and producing an aggregate `EvaluationReport`. |

## Base Classes (3)

| Type | Summary |
|:-----|:--------|
| [`AgentToolBase`](/docs/reference/wiki/agentic/agenttoolbase/) | Base class for tools that implements the common metadata plumbing, leaving only the behavior (`CancellationToken)`) for subclasses. |
| [`AiContent`](/docs/reference/wiki/agentic/aicontent/) | Base type for a single piece of content inside a `ChatMessage`. |
| [`ChatClientBase<T>`](/docs/reference/wiki/agentic/chatclientbase/) | Base class for `IChatClient` implementations that talk to an HTTP chat API. |

## Interfaces (19)

| Type | Summary |
|:-----|:--------|
| [`IAgentMemoryStore`](/docs/reference/wiki/agentic/iagentmemorystore/) | A long-term, cross-thread memory store: it remembers facts and can retrieve the ones most relevant to a query. |
| [`IAgentTool`](/docs/reference/wiki/agentic/iagenttool/) | An executable tool the model can call: a name, a description, a JSON-schema for its arguments, and an asynchronous invocation entry point. |
| [`IAgent<T>`](/docs/reference/wiki/agentic/iagent/) | A named, runnable agent: given a conversation, it produces a final answer (after optionally using tools and/or delegating to other agents along the way). |
| [`ICausalLanguageModel<T>`](/docs/reference/wiki/agentic/icausallanguagemodel/) | The minimal contract an in-process language model exposes to the local generation engine: given the tokens seen so far, produce the logits for the next token. |
| [`IChatClient<T>`](/docs/reference/wiki/agentic/ichatclient/) | A message-based chat model client that supports native tool calling, streaming, and structured output. |
| [`IChatInteractionStore`](/docs/reference/wiki/agentic/ichatinteractionstore/) | Stores recorded chat interactions (request key → response) so model calls can be deterministically replayed later without invoking any model. |
| [`IChatMiddleware`](/docs/reference/wiki/agentic/ichatmiddleware/) | A cross-cutting filter around chat-model calls (the AiDotNet analogue of Semantic Kernel filters / a middleware pipeline). |
| [`IChatPromptTemplate`](/docs/reference/wiki/agentic/ichatprompttemplate/) | Renders a chat conversation (a list of `ChatMessage`) into the single prompt string a local language model is fed. |
| [`IContentModerator`](/docs/reference/wiki/agentic/icontentmoderator/) | Checks whether a piece of text is safe/allowed — the seam guardrails use to screen agent inputs and outputs. |
| [`IConversationStore`](/docs/reference/wiki/agentic/iconversationstore/) | Persists multi-turn conversations keyed by a thread id, so an agent can remember earlier turns across separate runs. |
| [`IGenerationTokenizer`](/docs/reference/wiki/agentic/igenerationtokenizer/) | The minimal tokenizer contract the local generation engine needs: turn text into token ids, turn token ids back into text, and know which token marks end-of-sequence. |
| [`IGraphCheckpointer<TState>`](/docs/reference/wiki/agentic/igraphcheckpointer/) | Persists and retrieves `GraphCheckpoint`s, enabling durable resume and time-travel for graph runs. |
| [`IIncrementalCausalLanguageModel<T>`](/docs/reference/wiki/agentic/iincrementalcausallanguagemodel/) | An `ICausalLanguageModel` that supports incremental (KV-cached) decoding: it processes the prompt once, caches the per-position state, and then advances one token at a time without recomputing the whole sequence. |
| [`IMcpTransport`](/docs/reference/wiki/agentic/imcptransport/) | Carries Model Context Protocol (MCP) JSON-RPC requests to a server and returns the result. |
| [`INamedTensorSource`](/docs/reference/wiki/agentic/inamedtensorsource/) | A source of named weight tensors readable as `Double` arrays — the common surface over the safetensors and GGUF readers that `WeightImporter` imports from. |
| [`ITokenConstraint`](/docs/reference/wiki/agentic/itokenconstraint/) | Restricts which tokens the local engine may generate next, enabling *constrained decoding*: the model can only emit tokens the constraint permits, so the output is guaranteed to satisfy a structure (a fixed vocabulary, a grammar, a JSON sha… |
| [`IToolMiddleware`](/docs/reference/wiki/agentic/itoolmiddleware/) | A cross-cutting filter around tool/function execution (the AiDotNet analogue of Semantic Kernel's function-invocation filter). |
| [`ITrajectoryEvaluator`](/docs/reference/wiki/agentic/itrajectoryevaluator/) | Scores a captured `AgentTrajectory`, producing the reward signal the self-improving layer optimizes against (higher is better). |
| [`ITrajectoryStore`](/docs/reference/wiki/agentic/itrajectorystore/) | Stores captured `AgentTrajectory` records so the self-improving layer can replay, evaluate, and learn from past agent runs. |

## Enums (5)

| Type | Summary |
|:-----|:--------|
| [`ChatFinishReason`](/docs/reference/wiki/agentic/chatfinishreason/) | Describes why a chat model stopped generating a response. |
| [`ChatResponseFormatKind`](/docs/reference/wiki/agentic/chatresponseformatkind/) | Selects the shape the model's output must take (free text, arbitrary JSON, or schema-constrained JSON). |
| [`ChatRole`](/docs/reference/wiki/agentic/chatrole/) | Identifies who authored a `ChatMessage` in a conversation. |
| [`ImageMediaType`](/docs/reference/wiki/agentic/imagemediatype/) | The image formats accepted by multimodal chat models. |
| [`ToolChoiceMode`](/docs/reference/wiki/agentic/toolchoicemode/) | Controls whether and how a chat model is allowed to call tools on a given request. |

## Delegates (2)

| Type | Summary |
|:-----|:--------|
| [`ChatPipelineDelegate`](/docs/reference/wiki/agentic/chatpipelinedelegate/) | The delegate that invokes the next stage of a chat-middleware pipeline (ultimately the model call). |
| [`ToolPipelineDelegate`](/docs/reference/wiki/agentic/toolpipelinedelegate/) | The delegate that invokes the next stage of a tool-invocation pipeline (ultimately the tool itself). |

## Options & Configuration (9)

| Type | Summary |
|:-----|:--------|
| [`AgentExecutorOptions`](/docs/reference/wiki/agentic/agentexecutoroptions/) | Settings for an `AgentExecutor`: identity, system prompt, the tool-loop budget, and the sampling knobs forwarded to each model call. |
| [`ChatOptions`](/docs/reference/wiki/agentic/chatoptions/) | Per-request settings for a chat call: sampling controls, tool availability, and output format. |
| [`ContentSafetyOptions`](/docs/reference/wiki/agentic/contentsafetyoptions/) | Settings for `ContentSafetyMiddleware`: which sides to screen, what to say when blocking, and whether a violation throws or returns a refusal. |
| [`GraphRunOptions`](/docs/reference/wiki/agentic/graphrunoptions/) | Per-run settings for executing a `CompiledStateGraph`. |
| [`LocalEngineOptions`](/docs/reference/wiki/agentic/localengineoptions/) | Settings for `LocalEngineChatClient`: the reported model id, the default generation length, and the default sampling behavior (overridable per request via `ChatOptions`). |
| [`LocalSamplingOptions`](/docs/reference/wiki/agentic/localsamplingoptions/) | Controls how the next token is chosen from the model's logits: temperature, top-k, top-p (nucleus), and an optional seed for reproducibility. |
| [`MemoryAugmentationOptions`](/docs/reference/wiki/agentic/memoryaugmentationoptions/) | Settings for `MemoryAugmentedAgent`: how many memories to recall, the minimum relevance to include, and the heading used when injecting them into the conversation. |
| [`SupervisorOptions`](/docs/reference/wiki/agentic/supervisoroptions/) | Settings for a `SupervisorAgent`: its identity, an optional override of the routing system prompt, and the coordinator's loop/sampling budget. |
| [`SwarmOptions`](/docs/reference/wiki/agentic/swarmoptions/) | Settings for a `Swarm`: its identity and the overall step budget shared across all members for a single run. |

## Helpers & Utilities (16)

| Type | Summary |
|:-----|:--------|
| [`AgentGraphNodeExtensions`](/docs/reference/wiki/agentic/agentgraphnodeextensions/) | Bridges the agents layer and the graph runtime: adds an `IAgent` as a node in a `StateGraph`. |
| [`AgentToolFactory`](/docs/reference/wiki/agentic/agenttoolfactory/) | Creates `IAgentTool` instances from delegates or from objects whose methods are annotated with `AgentToolAttribute`. |
| [`AgenticTelemetry`](/docs/reference/wiki/agentic/agentictelemetry/) | The instrumentation source for the agentic subsystem: a named `ActivitySource` (traces) and `Meter` (metrics) following OpenTelemetry GenAI semantic conventions. |
| [`ChatClientExtensions`](/docs/reference/wiki/agentic/chatclientextensions/) | Convenience helpers over `IChatClient` for the common "prompt in, text out" case. |
| [`FineTuningDataConverter`](/docs/reference/wiki/agentic/finetuningdataconverter/) | Bridges the self-improving layer to the fine-tuning framework: converts a reward-filtered `FineTuningDataset` (prompt → good-completion pairs) into the framework's supervised `FineTuningData` shape so it can be handed to `SupervisedFineTuni… |
| [`GgufReader`](/docs/reference/wiki/agentic/ggufreader/) | Parses the GGUF weight format (used by llama.cpp and the wider GGML ecosystem) into a `GgufFile`: the header, the metadata key/value store, and the tensor directory. |
| [`GraphSpecialNodes`](/docs/reference/wiki/agentic/graphspecialnodes/) | Reserved node names recognized by the graph runtime. |
| [`ImageMediaTypeExtensions`](/docs/reference/wiki/agentic/imagemediatypeextensions/) | Conversions between `ImageMediaType` and its wire-format MIME string. |
| [`JsonSchemaGenerator`](/docs/reference/wiki/agentic/jsonschemagenerator/) | Generates JSON Schema (the dialect chat models consume for tool parameters and structured output) from .NET types and method parameters using reflection. |
| [`LoRAFineTuner`](/docs/reference/wiki/agentic/lorafinetuner/) | Runs the online self-improvement loop's final step: fine-tune (e.g. |
| [`MeaiChatClientExtensions`](/docs/reference/wiki/agentic/meaichatclientextensions/) | Fluent bridges between AiDotNet's `IChatClient` and Microsoft.Extensions.AI's `IChatClient`, in both directions, with full tool-calling support. |
| [`RewardFilteredDatasetBuilder`](/docs/reference/wiki/agentic/rewardfiltereddatasetbuilder/) | Builds a `FineTuningDataset` from captured trajectories by keeping only those whose reward meets a threshold and turning each into a (prompt, completion) pair — reward-filtered behavior cloning, the data-preparation half of online LoRA self… |
| [`SafetensorsReader`](/docs/reference/wiki/agentic/safetensorsreader/) | Parses the `` weight format — the safe, simple format used to distribute pretrained model weights — into a `SafetensorsFile`. |
| [`StructuredOutputExtensions`](/docs/reference/wiki/agentic/structuredoutputextensions/) | Helpers that turn a chat model's reply into a strongly-typed .NET object by constraining the model to a JSON schema derived from the target type and deserializing the result. |
| [`TextTensorDatasetConverter`](/docs/reference/wiki/agentic/texttensordatasetconverter/) | Tokenizes a text `FineTuningDataset` into tensor supervised-fine-tuning data so a tensor model (e.g. |
| [`WeightImporter`](/docs/reference/wiki/agentic/weightimporter/) | Imports named weight tensors from an `INamedTensorSource` (safetensors or GGUF) into an AiDotNet network's flat parameter vector. |

## Attributes (2)

| Type | Summary |
|:-----|:--------|
| [`AgentToolAttribute`](/docs/reference/wiki/agentic/agenttoolattribute/) | Marks a method as an agent tool so it can be discovered and exposed to a model with an auto-generated JSON schema. |
| [`ToolParameterAttribute`](/docs/reference/wiki/agentic/toolparameterattribute/) | Adds a description (and optional required-override) to a tool method parameter, enriching the auto-generated JSON schema. |

## Exceptions (4)

| Type | Summary |
|:-----|:--------|
| [`ContentSafetyException`](/docs/reference/wiki/agentic/contentsafetyexception/) | Thrown by `ContentSafetyMiddleware` when content is blocked and the middleware is configured to fail hard (`ThrowOnViolation`) rather than return a refusal. |
| [`GraphRecursionException`](/docs/reference/wiki/agentic/graphrecursionexception/) | Thrown when a graph run exceeds its configured maximum number of steps, which usually indicates a cycle that never reaches the end node. |
| [`HttpResponseException`](/docs/reference/wiki/agentic/httpresponseexception/) | An `HttpRequestException` that preserves the HTTP status code of a non-success API response on every target framework. |
| [`McpException`](/docs/reference/wiki/agentic/mcpexception/) | Thrown when an MCP server returns a JSON-RPC error or a malformed response. |

