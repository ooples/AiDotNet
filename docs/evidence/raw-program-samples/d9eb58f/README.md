# US-23 raw program measurement evidence

Implementation: `d9eb58f46e1f97c3bb300dea0c5e2b70f1321964`; explicit local core dependency:
`255feb24369702a32ea9db7a3f8a0b7a847d2762` (US-10, including the US-08 reused-learning guard).
No source changes separate these tested provider files from this evidence commit.

The full evolution integration project passed **963/963 tests on .NET 10 and .NET 8**,
zero skips. The actual .NET Framework 4.7.1 AiDotNet assembly passed **22/22 provider
tests**, using the archived small test project rather than source-linked replacement
provider types. The full legacy library build also passed, with 2,778 pre-existing
analyzer warnings and zero errors. This is not a claim that the entire AiDotNet test
suite or all 963 integration cases ran on .NET Framework.

`verification.zip` retains the six exact AiDotNet/core runtime DLLs, three changed
source files, test receipts, full build logs and the failed development attempts.
`integrity.json` and the internal manifest bind every artifact to its bytes.
All three final AiDotNet binaries report product version `0.204.0+d9eb58f46e1f97c3bb300dea0c5e2b70f1321964`.
The archive is diagnostic evidence, **not a published NuGet package** or a complete
standalone test installation; dependencies are restored through the source projects.

```powershell
python docs/evidence/raw-program-samples/d9eb58f/verify.py
$env:DOTNET_PROCESSOR_COUNT='4'
$env:DOTNET_gcServer='0'
$env:DOTNET_GCHeapHardLimit='0x300000000'
# Check out the pinned sources; substitute your absolute core project path.
dotnet test tests/AiDotNet.Evolution.Integration.Tests -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false -p:UseLocalEvolution=true -p:EvolutionProjectPath=<core-project> -p:GeneratePackageOnBuild=false
# Repeat with -f net8.0. Build src/AiDotNet.csproj with -f net471 and the same source-reference properties.
```

For the legacy subset, place the archived harness at
`.local/LegacyEvidenceTests/LegacyEvidenceTests.csproj`, then test it with the same
properties plus `-p:BuildProjectReferences=false` after building the legacy library.
The initial 4 GiB compiler cap exhausted memory; 12 GiB allowed the full build.
A source-linked diagnostic initially missed a Newtonsoft.Json runtime dependency,
and the first full build exposed ambiguous global JSON imports; both failed attempts
remain alongside their successful corrections. Neither preliminary result is substituted
for the final source-pinned integration receipts.

This verifies bounded raw scalar evidence retention, candidate/origin/provider binding,
mean and sample standard error, capacity/atomicity, corruption/freshness/force-fresh
behavior and current correctness rechecks. It cannot attest honest acquisition,
statistical independence, stationarity or other descriptors. The matched-prior noisy
campaign is in [core PR #62](https://github.com/ooples/AiDotNet.Evolution/pull/62).
Hosted CI, review, dependency integration and normal published-package verification
remain separate merge/release gates. No competitor or whole-roadmap completion claim.
