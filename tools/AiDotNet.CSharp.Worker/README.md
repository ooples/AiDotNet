# C# execution worker

Repository tooling for the `IProgramExecutionEngine` command boundary: real C# 12/Release compilation and console entry-point execution, without `dotnet-script`, project generation, SDK builds, restore or network requests during evaluation. Build once for the target runtime:

```powershell
dotnet build tools/AiDotNet.CSharp.Worker -c Release -f net8.0
dotnet tools/AiDotNet.CSharp.Worker/bin/Release/net8.0/AiDotNet.CSharp.Worker.dll --source candidate.cs --compile-only
```

Omit `--compile-only` to execute. Standard input/output are passed through. Supports ordinary and asynchronous console entry points and exit codes. Compile-only does not load the emitted assembly, execute module initializers, run analyzers or generators. Diagnostics contain only bounded compiler IDs and physical spans; exceptions have generic messages. Source is strict UTF-8, limited to 64 KiB characters/256 KiB bytes; emitted images are limited to 8 MiB. Compilation cancellation is cooperative, not a hard time or peak-memory guarantee.

Configure `ProgramSandboxOptions.Interpreters[ProgramLanguage.CSharp]` with an **absolute** `dotnet` executable and a trusted, appropriately quoted absolute worker DLL path followed by `--source {source}`. Set the compile template to the same command with `--compile-only`. Use a separate worker build for each target framework. Compilation references the worker's runtime assemblies; it does not establish compatibility with another deployment target or provide reference-package substitution.

The worker runtime configuration sets a 256 MiB managed GC heap ceiling, including compiler allocations. This does not cap native memory or total resident memory. It avoids sizing the GC heap from a large host when the supervisor supplies much smaller limits; supervisors must still provide enough address space for the CLR and enforce their own total-resource policy. Larger compiler workloads may need a separately built/configured worker and a correspondingly reviewed resource budget.

**This executable is not a security sandbox.** Running it directly gives candidate code the current account's filesystem and network permissions. The existing process runner adds time/output/concurrency limits and environment scrubbing, but is not filesystem/network isolation, and its memory enforcement can be unavailable or fail. For untrusted candidates, run the entire worker inside an independently provisioned least-privilege OS/container boundary, with no credentials, restricted mounts and network, and externally enforced resource limits. Never put sealed test answers, other runs, or sensitive host data within that boundary. Local regression tests execute only authored fixtures. Runtime measurements must separate compilation/startup from algorithm work and must not trust candidate-supplied timing or correctness claims.
