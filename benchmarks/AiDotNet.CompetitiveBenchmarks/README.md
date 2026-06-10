# AiDotNet Competitive Benchmarks

Head-to-head orchestration-overhead comparisons of AiDotNet's agentic runtime against **Semantic Kernel**
(.NET) and **LangGraph** (Python). All comparisons drive an identical deterministic in-process mock model (no
network, no LLM) with one tool registered, so the only variable measured is each framework's
orchestration/dispatch cost for a single tool-enabled turn — not model quality or provider latency.

This project is **benchmark-only**: it is never referenced by the shipped library or the serving host. It
pins Semantic Kernel inline (central package management disabled here) so a competitor framework never enters
AiDotNet's production dependency graph.

## Semantic Kernel (.NET) — live, runnable here

```bash
dotnet run -c Release --project benchmarks/AiDotNet.CompetitiveBenchmarks -- --filter *SemanticKernel*
```

Compares `AiDotNet.Agentic.Agents.AgentExecutor.RunAsync` against `Kernel.InvokePromptAsync`, both with one
tool. Uses BenchmarkDotNet's in-process toolchain (so it doesn't regenerate a harness project that would clash
with the repo's central package management).

**Representative result** (one machine, in-process mock model, single tool-enabled turn):

| Framework | Mean | Allocated |
|---|---:|---:|
| AiDotNet `AgentExecutor` | ~266 ns | ~944 B |
| Semantic Kernel `InvokePromptAsync` | ~2,718 ns (~10.2×) | ~4,458 B (~4.7×) |

AiDotNet's per-turn orchestration overhead is roughly an order of magnitude lower with ~4.7× fewer allocations.
(Absolute numbers vary by machine; the ratio is the takeaway.)

## LangGraph (Python) — cross-runtime via sidecar

BenchmarkDotNet cannot host a Python framework, so the LangGraph comparison times the same single tool-enabled
turn in both runtimes in comparable units (mean microseconds/iteration): AiDotNet in-process, and LangGraph via
the `langgraph_sidecar.py` subprocess.

```bash
# 1. Provision the Python side once:
pip install langgraph

# 2. Run the cross-runtime comparison (optional iteration count, default 2000):
dotnet run -c Release --project benchmarks/AiDotNet.CompetitiveBenchmarks -- --langgraph 2000
```

If Python or `langgraph` is not installed, the runner prints setup instructions and reports only the AiDotNet
number rather than failing. `langgraph_sidecar.py` can also be run standalone:
`python langgraph_sidecar.py 2000`.

## Notes

- The mock model returns a fixed answer so both frameworks do equal "model work"; the delta is pure framework
  machinery.
- For a fair LangGraph comparison the sidecar builds a minimal `StateGraph` with a single node, mirroring the
  one-turn scenario the .NET benchmarks measure.
- These are micro-benchmarks of dispatch overhead. End-to-end latency in real use is dominated by the model
  call; the point here is that AiDotNet adds far less overhead around it.
