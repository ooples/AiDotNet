# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning note.** Releases prior to 0.205.0 mapped fix/perf/refactor/docs
> commits to MINOR bumps instead of PATCH (a semver.org violation tracked as
> audit-2026-05 finding #18). Starting at 0.205.0 the bump rules follow
> Conventional Commits properly — see [.github/VERSIONING.md](.github/VERSIONING.md).
> Consumers pinning ranges across pre-0.205 releases should pin to exact versions.

> **History backfill.** Entries below were regenerated from `gh release list`
> by `scripts/regen-changelog.sh` for audit-2026-05 finding #15. Per-release
> summaries are the GitHub release title + commit count + date; for the
> per-commit detail, see the corresponding GitHub Release page linked from
> each version.

---

## [1.0.0](https://github.com/ooples/AiDotNet/compare/v0.231.0...v1.0.0) (2026-09-05)


### ⚠ BREAKING CHANGES

* remove the legacy flat-vector parameter component

### Features

* add deterministic quality-diversity evolution core ([34e05b6](https://github.com/ooples/AiDotNet/commit/34e05b6c1071f3053b58012e59c7b551d731fe47))
* add MAP-Elites AutoML strategy ([e3c7c4f](https://github.com/ooples/AiDotNet/commit/e3c7c4f2e68cddf0c0e6fd42d4bb853400499d5b))
* add non-mutating parameter observation and name oversized weights ([4d9b627](https://github.com/ooples/AiDotNet/commit/4d9b6279fa927d197d5c05c0fe0a6a2977fd2d6a))
* add streaming setparameterchunks and make the clone sweep flat-free ([fedc3f7](https://github.com/ooples/AiDotNet/commit/fedc3f783066217bf1aa0f0b066fc040cf0eb5a3))
* **agentic:** human-in-the-loop and subprocess models, panel judging, and repeat sampling ([3dca89a](https://github.com/ooples/AiDotNet/commit/3dca89a64bfd7fc4b10e2edb5ff287e71aa04921))
* **analyzers:** make an optimizer owning its own generator a build error ([94c3c74](https://github.com/ooples/AiDotNet/commit/94c3c74aa589e6c506f0698823b88823fc943ed0))
* **ci:** carry unchanged shards forward instead of re-instrumenting them ([8f548e6](https://github.com/ooples/AiDotNet/commit/8f548e6af2c7497496682a761533506f014b6916))
* **ci:** coverage-derived shard selection, tooling and proofs ([e64ba20](https://github.com/ooples/AiDotNet/commit/e64ba2008165463c0765ef3f3c320d91b4e9523f))
* **ci:** coverage-derived test selection to replace the unconditional 115-shard run ([3a8b668](https://github.com/ooples/AiDotNet/commit/3a8b668c2c35b804e6b790cc094d35e7939fe86f))
* **ci:** emit a per-shard test-impact digest artifact ([80d8bdf](https://github.com/ooples/AiDotNet/commit/80d8bdff6a8d378c2dc167f8f202a80cd60acefa))
* **ci:** let the nightly map cover heavy shards too ([38917d8](https://github.com/ooples/AiDotNet/commit/38917d84f2b50e6d9dcf52ba2046a7cb647b9921))
* **ci:** measure whether selection would have missed a failing shard ([4f58102](https://github.com/ooples/AiDotNet/commit/4f58102f5dbc08a8c0e9da11c6aaf1a7604de549))
* **ci:** ship selection in shadow mode, and widen the nightly window ([19415ab](https://github.com/ooples/AiDotNet/commit/19415ab040bb9187788a03c93ef680417183182f))
* **ci:** validate only the delta on master pushes ([6833575](https://github.com/ooples/AiDotNet/commit/68335753b3927fde125c8df1fac57f4f936acb0d))
* **ci:** wire shard selection into the test matrix ([282fa30](https://github.com/ooples/AiDotNet/commit/282fa30c6f0860551f2346008157953703e54bde))
* **clone:** automated, provably-correct cloning for options and layers ([b295ba3](https://github.com/ooples/AiDotNet/commit/b295ba3800d2ae93859f4fe993fd591db8736f38))
* complete the test-scale migration; no hand-written scaling remains ([97a7402](https://github.com/ooples/AiDotNet/commit/97a74027b19df9372ff5de1a47fc7d2e1556e0fa))
* **config:** make evolution configurable from a YAML file, secrets included ([e9910cb](https://github.com/ooples/AiDotNet/commit/e9910cb881cc2852920eac44c19ba7c8de20d51e))
* **evolution:** carry judge feedback into the next proposal ([97a9bc0](https://github.com/ooples/AiDotNet/commit/97a9bc0f14c660b75e2c65c5ed949048a068cf9a))
* **evolution:** early-stop on any evaluator metric, and grow descriptor ranges on demand ([d8456a8](https://github.com/ooples/AiDotNet/commit/d8456a86a435da9a38a28519de80499f2f27c9a0))
* **evolution:** embeddings, novelty distance, reasoning models, provenance, run outputs and metric ([1b9b7b6](https://github.com/ooples/AiDotNet/commit/1b9b7b63ea41fd5407d0f358d409eb595c2d76b5))
* **evolution:** facade entry points for evolution, program evolution, chat clients and the sandbox ([6841ea5](https://github.com/ooples/AiDotNet/commit/6841ea555ee0b065f81a9513d3be4b8dc114bad0))
* **evolution:** fill the three prompt sections the variation operator left empty ([66f006c](https://github.com/ooples/AiDotNet/commit/66f006cb8fc72f2972dcbbbff65875148ed0eeb9))
* **evolution:** finish changes-description mode with dual-target edit routing ([3f79932](https://github.com/ooples/AiDotNet/commit/3f79932cde09a1d028e7e9ada36d9bc1ce10b80e))
* **evolution:** make a finished run readable - per-candidate files and checkpoint ancestry ([e1b3c34](https://github.com/ooples/AiDotNet/commit/e1b3c34c91d847e03cab7f1af03a2e01344fb178))
* **evolution:** migration topologies, budget-tolerant resume and a directory checkpoint store ([f4d450d](https://github.com/ooples/AiDotNet/commit/f4d450d6a567d659e163af7cd59b42315947a4e9))
* **evolution:** openEvolve-parity selection, cascade, artifacts, stopping and tracing ([e93114b](https://github.com/ooples/AiDotNet/commit/e93114b545d2958c8c95a048ffd78da170b352a9))
* **evolution:** program-evolution substrate, prompt/LLM adapters and out-of-process sandbox ([c200ca0](https://github.com/ooples/AiDotNet/commit/c200ca025a10d5a08a385ddd7c196fd038143ab1))
* **evolution:** rank completed searches by any reported metric ([2395543](https://github.com/ooples/AiDotNet/commit/2395543a682c804bfcae1142d914d6902c02c293))
* **evolution:** rebase relative descriptors and remeasure the archive ([7001130](https://github.com/ooples/AiDotNet/commit/70011306168e188c9c81e2e4f0286a66c322e95d))
* **evolution:** recount earlier attempts, and make the novelty option reach the archive gate ([fdb41e2](https://github.com/ooples/AiDotNet/commit/fdb41e22517f08ff40dc422cbb1a491eedfe6967))
* **evolution:** refill workers as evaluations complete instead of at batch boundaries ([c9d8b45](https://github.com/ooples/AiDotNet/commit/c9d8b45d3318bf16b6bace744963a9ab1cc9ffa8))
* **evolution:** score an evaluator script that reports metrics instead of a quality ([69a724c](https://github.com/ooples/AiDotNet/commit/69a724cd0cab0484f3bdd2df3fafc1af07743404))
* **facade:** add documented Build data overloads ([23592a0](https://github.com/ooples/AiDotNet/commit/23592a0bef63792bee2815f27f43528a77146826))
* generate test-scale option bounds instead of hand-writing them ([bd4616c](https://github.com/ooples/AiDotNet/commit/bd4616c067b6d3e4ce87918b3517c6fd0c15ba89))
* **layers:** ask the following norm whether a bias is redundant ([940818c](https://github.com/ooples/AiDotNet/commit/940818c8a4f569a2ccc62dcf48f02270a799beb1))
* **layers:** make the convolution bias conditional rather than unconditional ([c8950e6](https://github.com/ooples/AiDotNet/commit/c8950e6fe9543b358e596cfa9eb2617d4891a329))
* migrate DenseNet too; no hand-written test-scale logic remains ([54acfb0](https://github.com/ooples/AiDotNet/commit/54acfb0e1765baf26d500cebdb02a13e09ff2f6d))
* migrate ResNet and EfficientNet test-scale config to the generator ([7bc06e9](https://github.com/ooples/AiDotNet/commit/7bc06e9ce6ee77f926a0676aea97f54cd7b9e1d2))
* **optimization:** LP/QP/assignment solvers + fix solver-shaped defects ([9d8303b](https://github.com/ooples/AiDotNet/commit/9d8303bf608c0430209b9c93010a7cbc70605ca9))
* **perf:** judge the census against a series, not a frozen point ([2e2b236](https://github.com/ooples/AiDotNet/commit/2e2b23652b08f3bc409df0bd634dae153208f4d0))
* **serialization:** one opt-in contract for layer construction state ([7594d78](https://github.com/ooples/AiDotNet/commit/7594d78f780727ba64d6f20315211a3b0da69fa7))
* support sparse tensors in the fast clone path ([19b5e91](https://github.com/ooples/AiDotNet/commit/19b5e9170c690c0bd04f8ac4beac155a61bbab55))
* synthesize bounds for constructor-only configurations ([cf85d1f](https://github.com/ooples/AiDotNet/commit/cf85d1fd2452994324846de7f0054c1f667e2421))
* **tools:** add a command-line front end for configuration-file evolution runs ([3318dee](https://github.com/ooples/AiDotNet/commit/3318deef7ed77676d299217b453bf3e3e9be5efe))


### Bug Fixes

* address hidden review findings ([205e59b](https://github.com/ooples/AiDotNet/commit/205e59bbb029813bc5bb7b389fb2a490082e3be3))
* address PR 2053 review findings ([7a86f01](https://github.com/ooples/AiDotNet/commit/7a86f01a622fc7e541f6355434222ff9a3a2b8ae))
* address the codeql review findings from pr 2052 ([2d46e91](https://github.com/ooples/AiDotNet/commit/2d46e91454b6164fa7d9329b076731844d2b038e))
* address the review findings on this branch ([01a39d8](https://github.com/ooples/AiDotNet/commit/01a39d8dee0f801332f008163dffe470aa1bed7a))
* address the second review round, and unbreak the net471 build ([d6dc75b](https://github.com/ooples/AiDotNet/commit/d6dc75ba0d6870f173436c67139bb251522ce473))
* **agentic:** make the subprocess chat client nameable from a configuration file ([37ce910](https://github.com/ooples/AiDotNet/commit/37ce9104eaabf4544976aa653b9cd9e539139bf2))
* align audio transformer optimizer recipes ([0a6f65b](https://github.com/ooples/AiDotNet/commit/0a6f65b03a91517b899387d53f9b3f9cea0c4ffa))
* align remaining paper training invariants ([3dc2afd](https://github.com/ooples/AiDotNet/commit/3dc2afd533d3478a61418ac94e33c78dccb2f20d))
* align risk and speech training recipes ([6f425b1](https://github.com/ooples/AiDotNet/commit/6f425b198ed8653da9c1036034a243d33e97113c))
* **audio:** replay GraFPrint dynamic graph during compiled training ([56c8e4a](https://github.com/ooples/AiDotNet/commit/56c8e4a117152bf8ec7c5bb4748d6ef3b2eb0358))
* **audio:** run voxlingua's own front end and output contract ([284cf3f](https://github.com/ooples/AiDotNet/commit/284cf3f536b0998d30912428621e283067d0b9bd))
* **audio:** size sed batch norm from the projection width ([3eb7fad](https://github.com/ooples/AiDotNet/commit/3eb7fad3fedfb3e613185fffee15ed7b54c3594c))
* **audio:** stop preprocessing twice in named-layer activations ([05dd99d](https://github.com/ooples/AiDotNet/commit/05dd99de1fd21f82fd502ce2684437bcd8d6d32e))
* **audio:** train the event detector on its paper's objective ([e940c2c](https://github.com/ooples/AiDotNet/commit/e940c2ccb2ba18eadf4121487ef5445369405b86))
* bound abandoned clone attempts so the sweep stops killing its own host ([f3cd8ef](https://github.com/ooples/AiDotNet/commit/f3cd8ef4c62e9ecb3034a8cc20cf9f9cf0681481))
* call the parameter-count rebind hook that was never wired up ([26122a2](https://github.com/ooples/AiDotNet/commit/26122a2a07bc32591e4a588f8b65fa4fbe861501))
* cap counts separately and never shrink inverse knobs ([1307dca](https://github.com/ooples/AiDotNet/commit/1307dca0b7588279160bdaecb2daea228dad16d2))
* **causal:** honour maxepochs as a maximum and name the lagrangian policy ([410ccd3](https://github.com/ooples/AiDotNet/commit/410ccd3868a8f3bd6fa585f4e7a688c69a1e88f2))
* **causal:** match the reference schedules for castle and amortizedcd ([93deefc](https://github.com/ooples/AiDotNet/commit/93deefc25ed7425b07d226836b11ecf8edb9bbc6))
* **causal:** restore CASTLE raw norm thresholding ([b245e49](https://github.com/ooples/AiDotNet/commit/b245e49f0825a79ef46bcfb20dea806651ef8cfe))
* centralize clone topology restoration ([feab62b](https://github.com/ooples/AiDotNet/commit/feab62beafd1566ef38d180b7fb04bea078e8310))
* centralize scheduler checkpoint recipes ([2998ace](https://github.com/ooples/AiDotNet/commit/2998ace212325ea71c4a984c0eea0468575363dc))
* **ci:** address hidden review findings ([e279c9e](https://github.com/ooples/AiDotNet/commit/e279c9ee69ba4f29dd711da294ba351458c184d4))
* **ci:** address PR 2029 review findings ([917b0e9](https://github.com/ooples/AiDotNet/commit/917b0e9be2a61ae93036e6f4f81c290d140723d0))
* **ci:** basename infra matching, atomic digest writes, merged carry ranges ([29a3844](https://github.com/ooples/AiDotNet/commit/29a38441b6adb834eb27631f3f491fa3a9946f2e))
* **ci:** close PR 2004 test regressions ([ad31da6](https://github.com/ooples/AiDotNet/commit/ad31da6d0041b071732982e3e5611d0af16e2405))
* **ci:** close PR 2026 acceptance regressions ([c1c2bc2](https://github.com/ooples/AiDotNet/commit/c1c2bc2e6e89e75c6018c0e8678c44ce73b96c7d))
* **ci:** close PR 2029 review gaps ([f9c55e8](https://github.com/ooples/AiDotNet/commit/f9c55e811344f1dd55765545a9f321cd50659f6e))
* **ci:** cover generated row-parallel tests ([e3089c3](https://github.com/ooples/AiDotNet/commit/e3089c360e33dac9a1555724cb8912bfd7c58de1))
* **ci:** cover generated vector model tests ([0544944](https://github.com/ooples/AiDotNet/commit/0544944188678224d59c57b3653d441980550d74))
* **ci:** declare the repvit-sam peak working set the same commit already moved ([dabaa89](https://github.com/ooples/AiDotNet/commit/dabaa897ecf61539ec6caef0c398b2e225469496))
* **ci:** declare the third repvit-sam memory metric the same commit moves ([3f00880](https://github.com/ooples/AiDotNet/commit/3f00880d17268641f02c627c7e163bf5afc0b538))
* **ci:** deleted files, script injection, stale audit map, missing timeout ([dc49ce6](https://github.com/ooples/AiDotNet/commit/dc49ce6bf841c8ddafd67d7e201e770cacdaa853))
* **ci:** emit flat range pairs from the digest as well ([7a17a42](https://github.com/ooples/AiDotNet/commit/7a17a42a8211fc2d89dab5386cc438e988e2d3a1))
* **ci:** enforce impact selection and exact-tree reuse ([b9ded76](https://github.com/ooples/AiDotNet/commit/b9ded765da3feaeafb086c87dbde6de3b1a39457))
* **ci:** escalate on any selection failure, and validate a manual run id ([55c2e86](https://github.com/ooples/AiDotNet/commit/55c2e8604f6fd32dd23c0abd0740709b083442fa))
* **ci:** escalate when the map cannot be read, not just when it is absent ([2442815](https://github.com/ooples/AiDotNet/commit/2442815289d0f43d659baf5407fd25af5b319bf8))
* **ci:** evaluate merge-queue reuse candidates first ([b8d19e7](https://github.com/ooples/AiDotNet/commit/b8d19e73e12730e622e0a557b00dfede672e5709))
* **ci:** fail safely across coverage-map lifecycle ([721cf66](https://github.com/ooples/AiDotNet/commit/721cf662ffedb51662d48ee22bf2bbbaae320aea))
* **ci:** flat range pairs and precedence, both found only by running the real path ([65e9440](https://github.com/ooples/AiDotNet/commit/65e9440604d8f09d2ca71efaed37aaa052f62028))
* **ci:** group the vi/vi alternation before excluding video ([ef6bf31](https://github.com/ooples/AiDotNet/commit/ef6bf31d8755d68c32aabd876073f56b05d90cfb))
* **ci:** harden the impact tooling against the adversarial review's findings ([cdd8f74](https://github.com/ooples/AiDotNet/commit/cdd8f748055207c9229588f8526cbdd15406f267))
* **ci:** keep draft events from cancelling ready validation ([2ae1923](https://github.com/ooples/AiDotNet/commit/2ae19231ee15ffba0fb26f53bcd4158f656e7d8f))
* **ci:** keep the regression gate and shard-coverage check honest under selection ([46cb6ba](https://github.com/ooples/AiDotNet/commit/46cb6ba576379007c1b4da1c1f3d5d7467d0826a))
* **ci:** let master reuse merge-queue validation, not only pr runs ([8ae7834](https://github.com/ooples/AiDotNet/commit/8ae78344948cbf43f5d10b86b38964120a58b9cb))
* **ci:** make impact selection and exact-tree reuse effective ([7f0807c](https://github.com/ooples/AiDotNet/commit/7f0807c4050d2613f7737de441141be5caa5e879))
* **ci:** preserve certified regression baselines ([0e4b5e2](https://github.com/ooples/AiDotNet/commit/0e4b5e2adc17a36017f2a4f064fb3dfced1722c1))
* **ci:** preserve failed build conclusions ([c43de37](https://github.com/ooples/AiDotNet/commit/c43de37ca5d6952e56d25210169c3c1aeb09aa30))
* **ci:** re-key the repvit-sam leases to the baseline the census now uses ([e873544](https://github.com/ooples/AiDotNet/commit/e8735442a5a436b1f3d92f3f98f9a77d30ce2bf2))
* **ci:** report lifecycle-aborted test shards ([b7f40e1](https://github.com/ooples/AiDotNet/commit/b7f40e1d919c1331eebf30eeb5bb90904c9049ca))
* **ci:** resolve final regression shards ([604f391](https://github.com/ooples/AiDotNet/commit/604f391047f994f5eb1edd06398930d915e95e7a))
* **ci:** restore compatibility and baseline validation ([2c81989](https://github.com/ooples/AiDotNet/commit/2c819890977290bdcd0e743ddca2e98f7e2b279a))
* **ci:** restore dependency and workflow validation ([84ab1ef](https://github.com/ooples/AiDotNet/commit/84ab1efe06415d59ce10321a084f1f8e18165e8d))
* **ci:** retry timed-out build artifact downloads ([0dac1dd](https://github.com/ooples/AiDotNet/commit/0dac1dd1866f3eccdedf9ef0f72def9ffa32fb24))
* **ci:** retry timed-out build artifact downloads ([024c561](https://github.com/ooples/AiDotNet/commit/024c561466297a6928e6c5ee815f86586ddbcc2b))
* **ci:** select against the map's coordinates, not the PR's ([49580f7](https://github.com/ooples/AiDotNet/commit/49580f72415a30047af3f7d00031475ed7a23365))
* **ci:** split always-run names defensively and count all runnable shards ([50737ae](https://github.com/ooples/AiDotNet/commit/50737ae0b3aaf93a82fc7473e77ca387e5dbc824))
* **ci:** stop collecting coverage on Integration C - Core ([1ecd634](https://github.com/ooples/AiDotNet/commit/1ecd6347c184aff44e901a048e72cdd751fd3fc7))
* **ci:** stop master duplicating pr validation, and video running twice ([af9058f](https://github.com/ooples/AiDotNet/commit/af9058f12e452dd4c8a4d557ca00745f762be4c2))
* **ci:** stop master duplicating pr validation, and video running twice ([228221e](https://github.com/ooples/AiDotNet/commit/228221e7170e783ca3092d1a4a1343e710b4bd06))
* **ci:** stop splitting always-run shard names on commas ([5745d55](https://github.com/ooples/AiDotNet/commit/5745d55b36e473dc9af2b91e89aa8437b1834fd8))
* **ci:** tolerate the map workflow not existing yet ([561361b](https://github.com/ooples/AiDotNet/commit/561361b5d88050a6422265ea2b79f8c55e74ab2a))
* **ci:** unblock PR validation ([3466865](https://github.com/ooples/AiDotNet/commit/34668656191bda347a4a630c639d4ecc805e5e8a))
* **ci:** verbatim directory creation and schema-gated map reads in the carry tool ([f59422b](https://github.com/ooples/AiDotNet/commit/f59422b1a3d5fcf42a065e5ed53681b2d3b7c3e4))
* **clone:** close generated lifecycle regressions with CI proof ([4a1d022](https://github.com/ooples/AiDotNet/commit/4a1d0228b2dbc7dc33075cf815e89c1f06d69a23))
* **clone:** complete automated clone acceptance criteria ([1a85906](https://github.com/ooples/AiDotNet/commit/1a85906d187fd42402fd30e0eb1b237db3d6bd36))
* **clone:** restore generated auxiliary state faithfully ([6114cf5](https://github.com/ooples/AiDotNet/commit/6114cf50df662f103fdfa9bb6860b9f41852c603))
* **clone:** restore generated continuation state ([3d6649e](https://github.com/ooples/AiDotNet/commit/3d6649ed18e45aea5138735bad24dbfebc9c9f57))
* **cloning:** rebuild read-only generic containers ([af38b3f](https://github.com/ooples/AiDotNet/commit/af38b3fb1222cea3d5b4a6cc99a416842e0b82f3))
* close PR 2004 clone and serialization regressions ([bf0d326](https://github.com/ooples/AiDotNet/commit/bf0d326ff1cee23bc2bacf2da295f07bfda5956c))
* **cmaes:** the step-size rule compared against the wrong reference length ([4303e99](https://github.com/ooples/AiDotNet/commit/4303e9932ce639ee9c80a8cf52fae93bb2d55ebe))
* compare clone parameter counts in the same materialization state ([c1419e8](https://github.com/ooples/AiDotNet/commit/c1419e8186baa23382e3a208d4c87234c7198dcb))
* complete clone state and RepViT invariants ([80bcb89](https://github.com/ooples/AiDotNet/commit/80bcb89c06ea624b33c6b16b0024fe6d5f3270eb))
* complete review follow-up invariants ([37773ed](https://github.com/ooples/AiDotNet/commit/37773ede175e341ec4e2861b3ec447ceefe8ff3c))
* consolidate Tacotron2 paper training path ([12e0e96](https://github.com/ooples/AiDotNet/commit/12e0e96e379d5c00fdf619504e85eaef171f1087))
* **conv:** apply the activation on the training path and stop grad mode choosing the kernel ([2739294](https://github.com/ooples/AiDotNet/commit/2739294ff0cfac551dcea82bbd5603bf767ce8fd))
* **conv:** preserve fused kernel parity by platform ([39985c9](https://github.com/ooples/AiDotNet/commit/39985c9f469104dc771d429f162406e8d153842f))
* **conv:** preserve platform-fast exact kernels ([f653eae](https://github.com/ooples/AiDotNet/commit/f653eaebb731691d079785527e43558211167a74))
* **daggnn:** stop orienting edges on the random seed and assert the recovered dag ([02a7676](https://github.com/ooples/AiDotNet/commit/02a767673e139263475c3f4729933c9d0e063071))
* **dcn:** normalize dcnv3 modulation across sample points and repair the ctor contract ([c3a5ca1](https://github.com/ooples/AiDotNet/commit/c3a5ca1f832a3bfa6b7447eee7fb29c670082a23))
* **deps:** align Entity Framework Core package train ([35adb27](https://github.com/ooples/AiDotNet/commit/35adb27a1429ddc79e8d2f46bbe0fda2e96c5f65))
* **diffusion:** declare PointE image generator conditional ([1428c34](https://github.com/ooples/AiDotNet/commit/1428c34cb0d9e715a6af27b1c5a2f1c98f679d25))
* **diffusion:** keep the conditioner in the chunk stream it was never registered in ([4b3a7fc](https://github.com/ooples/AiDotNet/commit/4b3a7fc856d8119c515f4d5b703b39aa689a43be))
* **diffusion:** preserve generated clone invariants ([beaaeda](https://github.com/ooples/AiDotNet/commit/beaaedaab5ab00f53b8cc5cf3c7bd97b5865f737))
* **diffusion:** stop an absent conditioner taking the whole parameter layout offline ([67f87a5](https://github.com/ooples/AiDotNet/commit/67f87a5330b731c1396271c2866a375da7d6002d))
* **diffusion:** stop an empty clone from materializing a foundation model ([acbd6aa](https://github.com/ooples/AiDotNet/commit/acbd6aac42cc86431a8626c4595d80750bc87eb3))
* **diffusion:** stream the component registry instead of a hand-listed parameter surface ([2941c5c](https://github.com/ooples/AiDotNet/commit/2941c5cd5354267a3c035d390865ea87204720b6))
* **diffusion:** stream the component registry instead of a hand-listed surface ([acff35d](https://github.com/ooples/AiDotNet/commit/acff35de11542f68d80b080ea296724820c1208d))
* distinguish a crash from a hang in the truncation report ([1992235](https://github.com/ooples/AiDotNet/commit/1992235a44b5e4adb0104580f20a9f63eaebd852))
* do not publish process dumps from a public repo ([a36baa9](https://github.com/ooples/AiDotNet/commit/a36baa9c2a6ab68eab06ce4b0ff8db53007a2e07))
* drop the null-forgiving operator from the conv stem fast path ([bc7727d](https://github.com/ooples/AiDotNet/commit/bc7727d174b201d9c16aa802b778e9cd18fe0b05))
* **embedding:** reject impossible deferred parameter payloads ([f761fbc](https://github.com/ooples/AiDotNet/commit/f761fbc67163a1f37cda53fb6fd4ef1263ab11c2))
* **evolution:** address security review findings ([9042649](https://github.com/ooples/AiDotNet/commit/90426497e518f33d50e07ba75769e6ee84d20c65))
* **evolution:** apply configured artifact retention ([c56defc](https://github.com/ooples/AiDotNet/commit/c56defcfc432b2cf2f7e545329cd417caeed5cf3))
* **evolution:** checkpoint variation-operator state ([c7e9be8](https://github.com/ooples/AiDotNet/commit/c7e9be831b0f458155582faefb7d3abbd0367910))
* **evolution:** classify SelectionPolicy as a derived option, not an unhashed one ([aba6feb](https://github.com/ooples/AiDotNet/commit/aba6feb3b9c56db9d3aa0d4f647bc1a662aea7d1))
* **evolution:** derive the archive grid from seed measurements ([c7d8a37](https://github.com/ooples/AiDotNet/commit/c7d8a377551a501610174f0871a6268d1c418e73))
* **evolution:** the defects an adversarial review of this branch found ([0d9e62a](https://github.com/ooples/AiDotNet/commit/0d9e62a5eac76265fa5a9a7dbd46e2c684cd8a13))
* **evolution:** wire orphaned subsystems and clone all engine settings ([ea72741](https://github.com/ooples/AiDotNet/commit/ea727415feb14c3ffa2e11c707f9923a601223d2))
* **evolution:** wire the artifact store, and refuse a store that can never receive anything ([0759993](https://github.com/ooples/AiDotNet/commit/0759993e663e7ba8c07c9c2dbea2ea8b15a21cba))
* **facade:** expose terminal builder methods on the interface ([c99f404](https://github.com/ooples/AiDotNet/commit/c99f404f3a5901b71e2bbff8b4b7c5b109a02353))
* fix post-2051 CI regressions without weakening invariants ([9e2d503](https://github.com/ooples/AiDotNet/commit/9e2d5031fb5d396a50435ebd0c29ae854299e9a6))
* fix remaining paper-faithful training invariants ([bf812ff](https://github.com/ooples/AiDotNet/commit/bf812ffc278d09ab6d3deeb556e9173c5bc55e8b))
* fix TOTO zero-warmup and DCGRU shape validation ([b2b0a58](https://github.com/ooples/AiDotNet/commit/b2b0a58895d25ca7688f9cb75eb8d075b3efafaf))
* **generator:** register a component slot declared through a plain interface ([f1e045b](https://github.com/ooples/AiDotNet/commit/f1e045b376489a082954c3dada2ae2cf72696eb8))
* **generators:** eliminate repeated semantic scans ([e11b664](https://github.com/ooples/AiDotNet/commit/e11b664e19a9c0b97e4c11e07a727d5f1ef80d1d))
* **generators:** emit clone factories for layers that forward ctor args to a base ([be4d880](https://github.com/ooples/AiDotNet/commit/be4d880d8041f9adf5bfcfe3221465d5e535a4bc))
* **generators:** give 15 more layers a working clone factory ([4aa6a2a](https://github.com/ooples/AiDotNet/commit/4aa6a2afdf95dbb8573b2fbcb47ee668cdaf97c9))
* **generators:** route the value kinds LayerStateBag already supported ([7c4cc65](https://github.com/ooples/AiDotNet/commit/7c4cc65b7dc4c9f32185523d50c0abaf98da454e))
* **graph:** prevent recursive training and arena-backed weights ([29ec59c](https://github.com/ooples/AiDotNet/commit/29ec59cc613b9e422dd5cd4a3cac147b97c8f77b))
* handle an unevolved NEAT population in parameter-count probes ([55fc991](https://github.com/ooples/AiDotNet/commit/55fc991ec0cb9e9a8333ec72e0010e69d82359f8))
* **inference:** refresh cached weights after parameter restore ([bc41891](https://github.com/ooples/AiDotNet/commit/bc41891a240d928dbee32cddb606acbff38da8ad))
* **inference:** register PagedCachedMultiHeadAttention's projection weights ([d0bca53](https://github.com/ooples/AiDotNet/commit/d0bca532a53daf371b2f274df2fe9c20caa0c4c5))
* **inference:** register the cached attention layers' projection weights ([fef2aa5](https://github.com/ooples/AiDotNet/commit/fef2aa5fe8a21154716329f27671aafef29281d6))
* initialize first compiled training lifecycle safely ([78749d2](https://github.com/ooples/AiDotNet/commit/78749d2969c1ee6019e496b052b7e9a5cc004b34))
* **init:** make parallel chunk arithmetic overflow-safe ([9f0aa9a](https://github.com/ooples/AiDotNet/commit/9f0aa9a1d8500492fdc2525896f623cb77d85a7b))
* **init:** make seeded parallel weight initialization machine-independent ([59e1b18](https://github.com/ooples/AiDotNet/commit/59e1b1828a706b76c29ff5b279d1667c7f6d1362))
* **init:** make seeded parallel weight initialization machine-independent ([f63ba64](https://github.com/ooples/AiDotNet/commit/f63ba64a78cefc6cdc34d39b9ce26bba6595adf3))
* **init:** stop a requested initialization seed from being silently discarded ([bdaa795](https://github.com/ooples/AiDotNet/commit/bdaa79578339bf14e061cc8d16c9dd75890f36ee))
* instrument construction and stop phase rows inheriting the previous model ([64cb7ae](https://github.com/ooples/AiDotNet/commit/64cb7ae29504635b86f84ba6d3cf34945748fb84))
* **jit:** close terminal trace retention race ([7a67382](https://github.com/ooples/AiDotNet/commit/7a67382aeb01c416ece8e95dc7ed2b1e721d7662))
* **jit:** make model capture fallback shape-safe ([2a8ca6b](https://github.com/ooples/AiDotNet/commit/2a8ca6b2da3e48b0af9e988dfe3b4a3344b17df2))
* **jit:** make model capture fallback shape-safe ([d7d50fb](https://github.com/ooples/AiDotNet/commit/d7d50fbd32dd77ad718167cbc26570d86bb1c89f))
* **jit:** release terminal stability traces ([7e51363](https://github.com/ooples/AiDotNet/commit/7e5136322f99a6a4db757e63c4097819235cb931))
* **jit:** validate replay on capture-compatible paths ([e97032b](https://github.com/ooples/AiDotNet/commit/e97032b7c9e1907be1a7bccef3ee780d95868455))
* keep clone construction manifests weight-free ([8689ac1](https://github.com/ooples/AiDotNet/commit/8689ac1dfc74e20edf99211586d22a7eb3ff87f0))
* keep the hang dump disabled, it failed 29 shards ([628c9a6](https://github.com/ooples/AiDotNet/commit/628c9a620f181a8c7aa9bb4d12603865754844d1))
* **layerhelper:** remove dead controllerinputsize store in the ntm factory ([c93f7ae](https://github.com/ooples/AiDotNet/commit/c93f7aea90d65d3025b8eb904915b7cdff6ef9e0))
* **layers:** materialize before enumerating a layer's trainable value slots ([933cc33](https://github.com/ooples/AiDotNet/commit/933cc330464328dcc0dc5f1ff1f39e752bda3bef))
* **layers:** materialize separable pointwise kernels ([3bf0490](https://github.com/ooples/AiDotNet/commit/3bf0490000021a498106d798cb061a66692300c5))
* **lstm:** re-infer resolved input width from parameter payload ([567f865](https://github.com/ooples/AiDotNet/commit/567f8655bbcc582a3334fd6862b261f92fee4de4))
* make reservoir initialization deterministic ([b87cfde](https://github.com/ooples/AiDotNet/commit/b87cfde3680f5f82368b81db7bf726098a762af4))
* make the bulk clone-buffer copy view-safe ([2c86812](https://github.com/ooples/AiDotNet/commit/2c8681237e53a3c1e80cc68e3605fad232be78c1))
* **mamba:** align Step/Predict on token ids, and stop dropping deferred layers ([9aa1a80](https://github.com/ooples/AiDotNet/commit/9aa1a8034d0c6ce88ce655d775b9a8b18a043155))
* match both sequence file spellings, not just the hang one ([4d54003](https://github.com/ooples/AiDotNet/commit/4d540035370cedf064e4e69484b897c1663b32d3))
* match count knobs by substring so head counts divide their width ([4e75a6e](https://github.com/ooples/AiDotNet/commit/4e75a6e513680c3edba05574afd4c888e508a350))
* match the sequence file vstest actually writes ([eb13d2d](https://github.com/ooples/AiDotNet/commit/eb13d2dc8c2425d4c1a928d7e347527261aa56f4))
* **mbpa:** return the k nearest nearest-first, and stop asking for a refused pack ([1a9b1e8](https://github.com/ooples/AiDotNet/commit/1a9b1e8ebcd47b67e75c193b5a3c377385c52b11))
* **nmf:** start from the data's singular vectors instead of an unseeded draw ([ee42954](https://github.com/ooples/AiDotNet/commit/ee42954cd639eb847cbc4d3ae75eddd7864a2bbe))
* **nmf:** stop the convergence tolerance from gating the random restarts ([1721275](https://github.com/ooples/AiDotNet/commit/17212751856f1c76d6b22c81357bb39aa59b0921))
* **nn:** carry both halves of the dnc controller state the same way ([a0bbf8f](https://github.com/ooples/AiDotNet/commit/a0bbf8f9552676efc3c02c920e8c37aa34250406))
* **nn:** give the dnc the recurrent controller its paper specifies ([3853a20](https://github.com/ooples/AiDotNet/commit/3853a2073e2ce7bf7b7f2ed5d75ecc897766e32f))
* **nn:** keep RecurrentGemma off the fused compiled training plan ([91e3cbb](https://github.com/ooples/AiDotNet/commit/91e3cbb22600606db47f0ab20b22241f8f9fe235))
* **ocr:** give abcnet the optimizer its published rate belongs to ([4ba640e](https://github.com/ooples/AiDotNet/commit/4ba640ea6ba2638423219bdaed7720a418045574))
* **ocr:** give svtr tps its released localization margin ([a06df29](https://github.com/ooples/AiDotNet/commit/a06df298c94bb69e7246c3e8cd74727205a0bc84))
* **optimizers:** adopt options through one method, and stop deserialize clobbering collaborators ([3b56c41](https://github.com/ooples/AiDotNet/commit/3b56c41e49920dbcec01571b38d05e909d437350))
* **optimizers:** honour options.seed on the model-based path ([7342cbf](https://github.com/ooples/AiDotNet/commit/7342cbfff79589007d34d55b91f871bcbd5fb15a))
* **optimizers:** honour options.seed, delete the duplicated options copy, and gate both ([6bb6b06](https://github.com/ooples/AiDotNet/commit/6bb6b065795285555d594d479f45e401ffc27e5f))
* **optimizers:** resolve optimizerbase by symbol, and assert collaborators by identity ([0cb4238](https://github.com/ooples/AiDotNet/commit/0cb42383a9b9c723b58638777da0f2a989621020))
* **parameters:** gate value-surface materialization on the declaration ([cace89f](https://github.com/ooples/AiDotNet/commit/cace89f61140e461fcb836fcc7d285427dadcb50))
* **parameters:** invalidate stale generated component snapshots ([b9bdfd6](https://github.com/ooples/AiDotNet/commit/b9bdfd61197fcd719a1dfc10c97127514f45d481))
* **parameters:** let AIDN082 see the chunked surface, and delete the 13 it was missing ([f27d2eb](https://github.com/ooples/AiDotNet/commit/f27d2eb6441a68f78a9a77c31d66c8d27fcc89ad))
* parse the sequence file vstest actually writes ([0aa7393](https://github.com/ooples/AiDotNet/commit/0aa739372593c20d3d18e2432059161dd8078bea))
* pass an absolute path for the diagnostics dump ([a6f03e6](https://github.com/ooples/AiDotNet/commit/a6f03e6b09d0a2eff506ba11350b52363c4e4954))
* **patchgan:** drop the bias on blocks whose batchnorm already shifts ([6cba715](https://github.com/ooples/AiDotNet/commit/6cba715be93c63df3a7ce4980663cbe5e0c2d444))
* **patchgan:** restore checkpoints written before the bias mode existed ([e7b34cf](https://github.com/ooples/AiDotNet/commit/e7b34cf35dd3509121125ad3c1502429876c67dc))
* **perf:** gr00tn1 grew a parameter slot it should always have had ([80d8f19](https://github.com/ooples/AiDotNet/commit/80d8f199df30053a8027ac1d65607c7838d2ab28))
* **performance:** cache echo state weight layouts ([f898c2f](https://github.com/ooples/AiDotNet/commit/f898c2ff7cc0449b1d66932c0e65a4e959005fc9))
* **perf:** stop a baseline drawn at a series low reporting a timing regression ([5b4add8](https://github.com/ooples/AiDotNet/commit/5b4add8a0aaf782227cd3d179d33d04f2fc07176))
* persist paper learning rate schedules ([3096997](https://github.com/ooples/AiDotNet/commit/3096997360c1d7f89e0185f82c4ce2eb71734772))
* **pointnet:** restore the gradient the neighbourhood max-pool was discarding ([ef68085](https://github.com/ooples/AiDotNet/commit/ef68085e8889fb3d0470cd00fea9b593e4b98018))
* post-2052 clone, serialization and CI-reuse failures ([969f198](https://github.com/ooples/AiDotNet/commit/969f198b986d4dc1d7a72d6d82ece9a34408942b))
* preserve generated model dimension invariants ([a5b39ba](https://github.com/ooples/AiDotNet/commit/a5b39ba0a59a3ab1a05052784b5bdd0d8eba99c9))
* preserve overlapping parameter ownership ([bee5df8](https://github.com/ooples/AiDotNet/commit/bee5df83bd8953ec29bf02fa8527b787afaf4fb4))
* preserve quantum state invariants deterministically ([a27bd00](https://github.com/ooples/AiDotNet/commit/a27bd00a911b964c92004c304eba287433dfa7c6))
* preserve scheduler checkpoint cadence ([d81bd80](https://github.com/ooples/AiDotNet/commit/d81bd801eee89a4fc205326c934f838c3cb9c120))
* preserve scheduler recipes and ONNX sessions ([b7f6e4b](https://github.com/ooples/AiDotNet/commit/b7f6e4bd9b0ebfbe4c18f17e8a810aef33dafd89))
* preserve strict paper training contracts ([2d039ba](https://github.com/ooples/AiDotNet/commit/2d039ba0fe02e1593f6e58a7d58dd9d5b75f1ebf))
* **pretrained:** address review — internal Inner + skip (not pass) fixture tests ([a1c1962](https://github.com/ooples/AiDotNet/commit/a1c196241af7fe1c809d423b516bcc27e26e0af1))
* **pretrained:** address second review pass ([f20203a](https://github.com/ooples/AiDotNet/commit/f20203aee54047cbe64f957967e2ee7027e3c33a))
* **pretrained:** permute q/k to interleaved RoPE layout for HF safetensors decoders ([e055374](https://github.com/ooples/AiDotNet/commit/e0553749e97b3e070f9200bb26cf3f7a7a4e90af))
* **pretrained:** permute q/k to interleaved RoPE layout for HF safetensors decoders ([c6de27e](https://github.com/ooples/AiDotNet/commit/c6de27eb5e7d278c31c8e0558556c8208e99d167))
* prioritize incomplete COW coverage diagnostics ([15dd50b](https://github.com/ooples/AiDotNet/commit/15dd50be0d97632d36537411bf0c27abc75c2b2c))
* **quality:** close tensor ownership findings ([4c9eb61](https://github.com/ooples/AiDotNet/commit/4c9eb61cc948a99cfee6214c0f586902d05d1dc7))
* **quantum:** preserve signed inputs as finite amplitudes ([dac183c](https://github.com/ooples/AiDotNet/commit/dac183c0791f5e2453da60c826b4dc457c52693f))
* **rbm:** dispose contrastive-divergence tensor input ([c5c5879](https://github.com/ooples/AiDotNet/commit/c5c58792df6872d6415e2e570f679048e6594b8b))
* **regularization:** a shrink factor that could invert the sign ([52710c6](https://github.com/ooples/AiDotNet/commit/52710c6d69cee227e32a48df3fc660d833abb3da))
* reject a parked restore shorter than the layer's known components ([2291a8b](https://github.com/ooples/AiDotNet/commit/2291a8bdc0be9469dc7db576c04e754d2959f7e7))
* remove the double hyphen that made Directory.Packages.props invalid xml ([d9d3813](https://github.com/ooples/AiDotNet/commit/d9d3813b2f5324235800117f1bc17278c60e1c0a))
* remove the stale batch-norm copy-on-write guard ([e86e0f7](https://github.com/ooples/AiDotNet/commit/e86e0f7ea0e16ba72444020a7638a5d5893ea8e3))
* reservoir was a tape dead end, aliased its state, and could build NaN weights ([4ac1547](https://github.com/ooples/AiDotNet/commit/4ac15479c425a9592dcdf3752d9077ae6cb68f27))
* resolve PR 2054 regression failures ([fb8f5f1](https://github.com/ooples/AiDotNet/commit/fb8f5f15de3ee6fff312e0b99b889733c5dfa38d))
* resolve remaining PR 2054 model regressions ([909a85f](https://github.com/ooples/AiDotNet/commit/909a85f4c0a9f6b50ba8e3c820cf4c2b19983c10))
* restore DAC recipe and DCRNN persistence ([2a10561](https://github.com/ooples/AiDotNet/commit/2a105617016e22314c92fe58190e9882b6d24e6d))
* restore Hamiltonian derivative training objective ([52d46eb](https://github.com/ooples/AiDotNet/commit/52d46eb20e45fe5f9f49e5f05a95b12e56f1c596))
* restore MedCLIP pretraining recipe ([c66dcb5](https://github.com/ooples/AiDotNet/commit/c66dcb57d933748a2468b8f59007a58508b52154))
* restore net471 invariant test compatibility ([ef54439](https://github.com/ooples/AiDotNet/commit/ef54439aa161f4649ecb903f6215831304f51312))
* restore net471 option copy compatibility ([7e5feb7](https://github.com/ooples/AiDotNet/commit/7e5feb70da44cde2bb39b98ea7ca13d23b7724bb))
* restore paper DCRNN recurrent diffusion ([cb9ecf5](https://github.com/ooples/AiDotNet/commit/cb9ecf59db264e4c80876a971ee05edee0906c62))
* restore paper optimizer and FiNS training behavior ([9b35c1e](https://github.com/ooples/AiDotNet/commit/9b35c1ebd5d990403dfdc95076fa3cd49ecd7e8a))
* restore paper-faithful Deep KWS architecture ([e1b634d](https://github.com/ooples/AiDotNet/commit/e1b634d4dc893c97c9e9ca78b34ed7bcc25129b3))
* restore remaining paper training invariants ([5ed6c77](https://github.com/ooples/AiDotNet/commit/5ed6c772f5bc7d6f11459422bf1a2af44821795b))
* restore Squeezeformer extended Noam schedule ([5796503](https://github.com/ooples/AiDotNet/commit/57965038226f071facba5fafca52405a46060d0a))
* **review:** address generator and tensor lifecycle findings ([9fec033](https://github.com/ooples/AiDotNet/commit/9fec03343893c625e3a68d215b55e3d9f79a661d))
* **rg-lru:** restore finite fused training for Griffin, Hawk, and RecurrentGemma ([2d60bbe](https://github.com/ooples/AiDotNet/commit/2d60bbeecabcba3a6e679f577b9e78a49885d684))
* **rl:** train the deep q-network on its published rmsprop settings ([422a92a](https://github.com/ooples/AiDotNet/commit/422a92a90ce9f6ba8bc628df28f768742cb2911c))
* **robotics:** build helix's optimizer from its own training options ([61f0518](https://github.com/ooples/AiDotNet/commit/61f05182dc6deb299cfd5a73b53e2f980c18d67e))
* scale declared integers proportionally instead of matching names ([59080d7](https://github.com/ooples/AiDotNet/commit/59080d706db3d3fcf01b2c3a581fe1eae1f7a3a5))
* scope the auto-compile test override to the calling flow ([d36c6ec](https://github.com/ooples/AiDotNet/commit/d36c6eca178552f7de40ccf24197e19a072cad01))
* **segmentation:** train repvitsam at its family's rate, not the generic default ([3cbd616](https://github.com/ooples/AiDotNet/commit/3cbd6166dad52c0cce257bb417efab5cbbc49a35))
* **serialization:** carry a component's configuration, not just its type ([66fec3a](https://github.com/ooples/AiDotNet/commit/66fec3ad0acb8f915150ab35c00874244d2f3089))
* **serialization:** let a restored expression call the framework's own maths ([e60651b](https://github.com/ooples/AiDotNet/commit/e60651baf0a51cfbec6311692b0886066377c195))
* **serialization:** write the resolved input shape for every layout node ([726e2ec](https://github.com/ooples/AiDotNet/commit/726e2ec20ae5f4d782e101f3b7dcaa5cc13d6f96))
* **som:** Predict passthrough; plus generator symbol-retention hygiene (no perf win) ([#2036](https://github.com/ooples/AiDotNet/issues/2036)) ([b7087b3](https://github.com/ooples/AiDotNet/commit/b7087b3f7c3a1ecffd84579623f6f46a7f0ac0c8))
* sparse tensors cannot be copy-on-write shared ([50bde63](https://github.com/ooples/AiDotNet/commit/50bde638f7aeeae0047f02891c4b6bb711c21278))
* **ssm:** construction-sized parameter surfaces, RWKV7 token contract, D protection ([af0e1c2](https://github.com/ooples/AiDotNet/commit/af0e1c24229f0a2a9dc588bd3ee83b5ac3482868))
* stabilize remaining non-clone training invariants ([f80e4e6](https://github.com/ooples/AiDotNet/commit/f80e4e6eb593ddc1517b792fae9633371d5e37c5))
* **state:** rebind derived views after parameter restore ([3496f8b](https://github.com/ooples/AiDotNet/commit/3496f8bb833a5956d038547d233a394ad98f9e2b))
* stop allocating full-size scratch activation caches during conv init ([2214792](https://github.com/ooples/AiDotNet/commit/2214792da82cb7323310fb74efd93c15d0400e83))
* stop the clone sweep leaking two materialized models per timeout ([5d5fa6f](https://github.com/ooples/AiDotNet/commit/5d5fa6f55d4e479a5f60ba064273655858feecdb))
* stop the generator emitting an empty table into every compilation ([d53a7f4](https://github.com/ooples/AiDotNet/commit/d53a7f4642685a8c2b97ec24ada8b6403aa70da8))
* stop the GPU quantum prescale returning NaN for every all-zero row ([08b2b31](https://github.com/ooples/AiDotNet/commit/08b2b3169f1d28b20f87af75b6c0ecf7800307a6))
* stream the clone sweep independence check through parameter chunks ([efa475d](https://github.com/ooples/AiDotNet/commit/efa475db077d15557b6c6cf34909734dcaa257c7))
* **streaming:** honor overridden parameter count during auto-detection ([5c0a7ba](https://github.com/ooples/AiDotNet/commit/5c0a7ba9ec87d0e17c20ff6961081112984d23e5))
* **tests:** align MGTSD architecture with forecast options ([18e72a7](https://github.com/ooples/AiDotNet/commit/18e72a7ade424954d3c2c7d6b08db64b3fba7055))
* **tests:** align RecurrentGemma policy guard ([4d021a6](https://github.com/ooples/AiDotNet/commit/4d021a62283278d6bc47ee2348c629fc378f0710))
* **tests:** bound codec TTS clone sweep options ([dbdb3b8](https://github.com/ooples/AiDotNet/commit/dbdb3b8d310d4e75b029ea33462100e96e765f07))
* **tests:** bound exhaustive layer clone sweep ([3edc9c1](https://github.com/ooples/AiDotNet/commit/3edc9c1e34fadda7edd30b2601ba6fff42552c42))
* **tests:** bound Nemotron clone sweep options ([90d82a7](https://github.com/ooples/AiDotNet/commit/90d82a78a414ee904eecf7150210276b24406bfb))
* **tests:** classify every clone attempt and keep coverage off the timing shards ([68551cb](https://github.com/ooples/AiDotNet/commit/68551cbc235151400ce0e6bd6e308767daff5493))
* **tests:** compare SeACo shape dimensions portably ([c7a2529](https://github.com/ooples/AiDotNet/commit/c7a2529a8ed6a16643d40aed3f2baccafadb292d))
* **tests:** drop duplicate provideslearnableshift left by the rebase ([1064ae2](https://github.com/ooples/AiDotNet/commit/1064ae2ee7d34db77e32657542ffb123077a01fd))
* **tests:** give the training-error invariant enough steps to be true at paper learning rates ([733b17e](https://github.com/ooples/AiDotNet/commit/733b17ed32a819af7dbb9618d878aa4ac66141cc))
* **tests:** implement provideslearnableshift exactly once per ilayer double ([796c79a](https://github.com/ooples/AiDotNet/commit/796c79ad4b4d0e102522cd208aabfef42731d7e0))
* **tests:** implement provideslearnableshift on the two hand-rolled ilayer mocks ([8669d1d](https://github.com/ooples/AiDotNet/commit/8669d1d40244b847316de7002012210ec474ed42))
* **tests:** keep the RecurrentGemma regression asserts net471-compatible ([822a244](https://github.com/ooples/AiDotNet/commit/822a244efa62613b9b79dc5bfc977f1abe2a3823))
* **tests:** let Dessurt clear the AdamW transient ([8583b8a](https://github.com/ooples/AiDotNet/commit/8583b8ab03f5b698b3b6d9914ef980acdfbbaaca))
* **tests:** measure the train-vs-test invariant on the claim it makes ([5e5c8dc](https://github.com/ooples/AiDotNet/commit/5e5c8dc9dba1cc3e43507e9602ca48e380cd2fb4))
* **tests:** prevent clone sweep model accumulation ([f5a331d](https://github.com/ooples/AiDotNet/commit/f5a331d461de2cf699c9e3e76520263bbfc1a7bb))
* **tests:** read the optimizer options as field OR property after the refactor ([c0eef9a](https://github.com/ooples/AiDotNet/commit/c0eef9afc327cf6ed8430c27a90862c58fc7e0d6))
* **tests:** repair two harness defects that failed before asserting ([#2035](https://github.com/ooples/AiDotNet/issues/2035)) ([1f3bc96](https://github.com/ooples/AiDotNet/commit/1f3bc962caf22f4e707c67867a4786817a0976a9))
* **tests:** support legacy echo state allocation checks ([e021019](https://github.com/ooples/AiDotNet/commit/e021019bfe7052cc8852dca0f9082af9e4b50aed))
* **tests:** support net471 finite checks ([0a9cc96](https://github.com/ooples/AiDotNet/commit/0a9cc967d030d3c82d9709804647a1b846e1d7db))
* **tests:** support net471 finite gradient assertion ([63ed6ec](https://github.com/ooples/AiDotNet/commit/63ed6ecee1b9ff6f2f31f310d950311199196fd7))
* **tests:** use a finite check that exists on net471 ([a31a900](https://github.com/ooples/AiDotNet/commit/a31a900db465c60ff11319e93d7d8141e704e082))
* **timegan:** restore recurrent sequence architecture ([5b6cf58](https://github.com/ooples/AiDotNet/commit/5b6cf58187cb4602d1042d0cdf2e02143662a084))
* **training:** address PR 2029 review findings ([6ec49d2](https://github.com/ooples/AiDotNet/commit/6ec49d204e51b46ae5b951d3d5be5f41bc33ec14))
* **training:** align strict invariants with paper objectives ([52a9874](https://github.com/ooples/AiDotNet/commit/52a9874576c415d9946135f26994f20669190617))
* **training:** align strict invariants with paper objectives ([bd7f232](https://github.com/ooples/AiDotNet/commit/bd7f23246acf2dd08e9a4d2a1ae0288c862ec936))
* **training:** complete paper-faithful objective fixes ([d6e45d1](https://github.com/ooples/AiDotNet/commit/d6e45d1c445fc37cb73a65ff11083a4fcddafe34))
* **training:** drive callbacks from the image-space epoch loop ([e9b7f2f](https://github.com/ooples/AiDotNet/commit/e9b7f2ff9114bc6efda0880ba69afa9beb51702e))
* **training:** flatten gradients in the canonical parameter order ([2cc61c9](https://github.com/ooples/AiDotNet/commit/2cc61c97978df788b51a64d5c1a1fc9f9c8fa6d8))
* **training:** honor partial-freeze selection in fused compiled path ([012adcd](https://github.com/ooples/AiDotNet/commit/012adcd95206df5b63f62ca36194be0884a6177d))
* **training:** preserve partial-freeze invariants in fused plans ([51f04de](https://github.com/ooples/AiDotNet/commit/51f04de942d1d7453d85fcd40e9585d34b915c64))
* **training:** record RG-LRU in compiled plans ([69896e7](https://github.com/ooples/AiDotNet/commit/69896e77ac088e36a9c63f9a4ec05a098176a2f3))
* **training:** repair PR 2032 regressions ([ff62eed](https://github.com/ooples/AiDotNet/commit/ff62eedad8fa29b5aad1396729d3e3b021b62048))
* **training:** reset the optimizer state a training step actually uses ([edec97a](https://github.com/ooples/AiDotNet/commit/edec97a9d0cd0d4aa9a8ad87c5d5b9bc5df8dfeb))
* **training:** resolve cross-target review regressions ([df22056](https://github.com/ooples/AiDotNet/commit/df22056707fa7b0c383f083eb70d4c8e879afe62))
* **tts:** measure dittotts memorization past its first adam update ([92def7e](https://github.com/ooples/AiDotNet/commit/92def7ec97dd6af5283863dfe65a0a552976ea7e))
* **tts:** train OpenVoiceV2 at the learning rate its options declare ([27fc29b](https://github.com/ooples/AiDotNet/commit/27fc29bc4f4be9e3da67fd1425815a7f63adc303))
* **tts:** train the vits family on its published adamw recipe ([2d1a248](https://github.com/ooples/AiDotNet/commit/2d1a2481c45ddbf2421d677be4ae193fc79289d9))
* wire the isolated probe in, and correct two defects the first version shipped ([ae7406f](https://github.com/ooples/AiDotNet/commit/ae7406f71a7d07517a9df67ab10fb2700e9b83b0))


### Performance

* bulk-copy clone buffers and copy tensors before resolving shape ([cb548a7](https://github.com/ooples/AiDotNet/commit/cb548a70301cec99b1ed39a1afc3a1a38df51e44))
* fill conv kernels in chunks instead of a full-size staging array ([e23dfeb](https://github.com/ooples/AiDotNet/commit/e23dfebbe2206decfaa778b4dd30b3a1cfcb1420))
* **internimage:** use dcnv3 linear offset/mask projections as the paper specifies ([f76e12c](https://github.com/ooples/AiDotNet/commit/f76e12c259dbaf57916e0abfab11b94bcdf1d642))
* **layers:** stop re-copying parameters once per level of nesting on write ([9cd287e](https://github.com/ooples/AiDotNet/commit/9cd287e4cdf15d9754a523c0c04ec6ecea9f1ed9))
* own activation-cache lifetime in the base class, PyTorch-style ([1c9ff26](https://github.com/ooples/AiDotNet/commit/1c9ff26050ad349b67c28321f3cf83e4230b6b62))
* resolve a clone's shapes without materialising the activation ([dc22d62](https://github.com/ooples/AiDotNet/commit/dc22d62a519c430ffdb6131e4db84bbb024f73b9))
* **serialization:** work out a component's configuration once per type ([96a120f](https://github.com/ooples/AiDotNet/commit/96a120f819759ba28fa4f66b5f7a731c410865cb))
* stop the VAE stack pinning every activation for the whole forward ([d67773f](https://github.com/ooples/AiDotNet/commit/d67773f6dd3bcbd1fa58bc0b5341e6d3cb08e90e))
* **tests:** stop the loss-domain guard allocating per element ([e33cf39](https://github.com/ooples/AiDotNet/commit/e33cf39cdfc70e018fcee843d01c9fd3d8d7adbc))
* **training:** skip unused fused selection walk ([0a16ea8](https://github.com/ooples/AiDotNet/commit/0a16ea846b9f4511423d8b585ba1dade8ae5d76b))


### Reverts

* drop my duplicate provideslearnableshift on the two ilayer mocks ([feda8d3](https://github.com/ooples/AiDotNet/commit/feda8d3d2e61643e6e855e098710d86b19e01d61))
* remove invariant-weakening fixture changes ([fe7440a](https://github.com/ooples/AiDotNet/commit/fe7440a0947f61418ed658bf37d01a1dd84c84a3))


### Refactoring

* **clone:** enforce generated model lifecycle ([de55b75](https://github.com/ooples/AiDotNet/commit/de55b75e05bda7df2280a61519b5499af78b6fc9))
* **evolution:** one public type per file in conventional folders ([745d8d0](https://github.com/ooples/AiDotNet/commit/745d8d082d49fd2eaa2aa4865ec91731c814c9de))
* extract the duplicated nndsvd entry in nmf initialization ([08a8032](https://github.com/ooples/AiDotNet/commit/08a803214489fa1324cca4f4fc1e8f864685e79c))
* **optimizers:** delete the cached typed options copy, 42 updateoptions overrides to 6 ([c687c6f](https://github.com/ooples/AiDotNet/commit/c687c6fd659f769dfad2052129684ea0410c8b6b))
* **optimizers:** move seeding wholly into the base instead of per optimizer ([fc11d5e](https://github.com/ooples/AiDotNet/commit/fc11d5e1c3f76f477e5f12e02b3a09bc6c15cb5b))
* remove the legacy flat-vector parameter component ([a3114b1](https://github.com/ooples/AiDotNet/commit/a3114b1cb8c6d42ef0f2f42fce818e156b141810))
* replace the cap tiers with one proportional rule ([ea32351](https://github.com/ooples/AiDotNet/commit/ea32351c7251f673baa30a5765934477f96b0667))


### Build System

* stop packing six projects nothing publishes or consumes ([d8dab69](https://github.com/ooples/AiDotNet/commit/d8dab6908d7b51ed153148b158899ad988250df3))
* tests/AiDotNet.Tests succeeds on net10.0. ([647fb7e](https://github.com/ooples/AiDotNet/commit/647fb7ebed2704c67c0120780887c94ef4010345))
* **tests:** target net8.0, which the library ships but nothing compiled ([bde2742](https://github.com/ooples/AiDotNet/commit/bde274229903ea7d005f12a1086fa794214d5d58))


### Documentation

* attach the sed constructor documentation to its own constructor ([fa4adc5](https://github.com/ooples/AiDotNet/commit/fa4adc5a253f918e373f0172012eeaed569357a8))
* **ci:** record the map builder's measured behaviour at full scale ([0400c8f](https://github.com/ooples/AiDotNet/commit/0400c8f91bb91164e1afd8f8fa823918c42ea2bc))
* **evolution:** correct elite remeasurement behavior ([1847bb4](https://github.com/ooples/AiDotNet/commit/1847bb43a83a62f9a39403e3c02fad91cd8af080))
* **evolution:** for Beginners remarks + remove null-forgiving operators ([245439b](https://github.com/ooples/AiDotNet/commit/245439b8b32fb1c9acf4bc90c060cc1137c2e78c))

## [Unreleased]

### Breaking Changes

* `TOTEMOptions<T>.CommitmentWeight` was removed from the forecasting configuration. TOTEM forecasting now consumes a separately trained, frozen tokenizer/codebook; configure commitment loss in that tokenizer-pretraining workflow rather than on the forecasting model.
* `QuantileRegressionOptions<T>.LearningRate` was removed because quantile regression now uses an exact linear-program solver rather than gradient descent; there is no learning-rate replacement. `QuantileRegressionOptions<T>.MaxIterations` moved to `SolverOptions.MaxIterations`, alongside the other simplex controls.
* `NeuralNetworkRegressionOptions<T, TInput, TOutput>.Optimizer` was replaced by `OptimizerFactory`. The factory receives the model it will optimize and creates one optimizer per model, preventing clones from sharing mutable optimizer state.
* `IAiModelBuilder<T, TInput, TOutput>` now declares `ConfigureSegmentationVisualization`. External implementations of the public interface must add the method; implementations can forward the supplied configuration to their result defaults or return `this` after storing it for their renderer.

---

## [0.231.0](https://github.com/ooples/AiDotNet/compare/v0.230.0...v0.231.0) (2026-07-17)


### Features

* **#1209,#1214:** lazy shape inference + architecture-optional NeuralNetworkBase ([#1259](https://github.com/ooples/AiDotNet/issues/1259)) ([03916e7](https://github.com/ooples/AiDotNet/commit/03916e715e0f74e0e97a5360fa7776eba5b9a377))
* **#1211:** ONNX symbolic-axis end-to-end with ONNX Runtime ([#1269](https://github.com/ooples/AiDotNet/issues/1269)) ([66a966e](https://github.com/ooples/AiDotNet/commit/66a966e9bbecdff46bf273fc6ed187f56d52b1af))
* **#1213:** RBMLayer lazy ctor — first composite-layer reference impl ([#1243](https://github.com/ooples/AiDotNet/issues/1243)) ([7458d38](https://github.com/ooples/AiDotNet/commit/7458d387f836b7d6615bca9a1257cbe621ad55fd))
* **#1237:** widen ParameterCount int → long across base classes + chunked GetParameterChunks API ([#1244](https://github.com/ooples/AiDotNet/issues/1244)) ([25c7918](https://github.com/ooples/AiDotNet/commit/25c791822669c0694ee9529a4b798f922ec428b7))
* **#1239:** scored ctor matcher + migrate throw sites to MissingLayerCtorException ([#1246](https://github.com/ooples/AiDotNet/issues/1246)) ([0bb1461](https://github.com/ooples/AiDotNet/commit/0bb1461cc7a14bb2150b03851e655ad2747a37bd))
* **#1273:** true-async + compile-host adoption across diffusion / VAE / VLM / LoRA ([#1279](https://github.com/ooples/AiDotNet/issues/1279)) ([1554543](https://github.com/ooples/AiDotNet/commit/1554543bc03a9efeb73602c2ee669f4582a142f5))
* **#1276:** 30 dataset loaders + 113 integration tests (text/LM, LLM-eval, vision, audio) ([#1277](https://github.com/ooples/AiDotNet/issues/1277)) ([b79dfa6](https://github.com/ooples/AiDotNet/commit/b79dfa6584046264fcdf23aecf1be07ca78b5784))
* **#1342:** int8 weight-quantization inference surface (internal + InternalsVisibleTo) ([#1348](https://github.com/ooples/AiDotNet/issues/1348)) ([f111f45](https://github.com/ooples/AiDotNet/commit/f111f45b915f130126696eb9d208fbb60067ac71))
* **#1370:** shape oracle TryDeclareShape() — eliminate LoRA warmup forward when ctor carries enough info ([#1388](https://github.com/ooples/AiDotNet/issues/1388)) ([18a3d6a](https://github.com/ooples/AiDotNet/commit/18a3d6a999fda0e6bd578409c8309733df8a9bd8))
* **#1650/#642:** diffusion CUDA-graph capture for eval-mode UNet inference (~3.2x) ([#1650](https://github.com/ooples/AiDotNet/issues/1650)) ([64a8c1f](https://github.com/ooples/AiDotNet/commit/64a8c1f765986a58f95b1f77caae029905e7379b))
* Add Google Speech Commands v2 data loader ([#1135](https://github.com/ooples/AiDotNet/issues/1135)) ([350bc33](https://github.com/ooples/AiDotNet/commit/350bc3327390a6666fc8cd5596cea334ce21815f))
* add NCHW layout to Cifar100DataLoader + EuroSatDataLoader ([d2f5bff](https://github.com/ooples/AiDotNet/commit/d2f5bffe9dd31a92a5d3e6db1f9fa41ca2f15d12))
* **agentic:** Phase 0 — IChatClient&lt;T&gt; model abstraction (epic [#1544](https://github.com/ooples/AiDotNet/issues/1544)) ([#1545](https://github.com/ooples/AiDotNet/issues/1545)) ([29e1bea](https://github.com/ooples/AiDotNet/commit/29e1bead92feb26283c6eff1fdc6028c5925688d))
* **agentic:** Phase 1 — typed durable graph runtime (epic [#1544](https://github.com/ooples/AiDotNet/issues/1544)) ([#1548](https://github.com/ooples/AiDotNet/issues/1548)) ([2a39306](https://github.com/ooples/AiDotNet/commit/2a393061090a8aba109a6f5f4ba3c119d61d5b3e))
* **agentic:** Phase 2 — multi-agent orchestration (epic [#1544](https://github.com/ooples/AiDotNet/issues/1544)) ([#1551](https://github.com/ooples/AiDotNet/issues/1551)) ([1250ab0](https://github.com/ooples/AiDotNet/commit/1250ab031b3d731dea39e4690835947aca49ecb6))
* **agentic:** Phase 3 — local in-process inference (epic [#1544](https://github.com/ooples/AiDotNet/issues/1544)) ([#1552](https://github.com/ooples/AiDotNet/issues/1552)) ([5c37400](https://github.com/ooples/AiDotNet/commit/5c374005d6fc60fd792b72d4d35f6336c42f71dd))
* **agentic:** Phase 4 — self-improving orchestration (epic [#1544](https://github.com/ooples/AiDotNet/issues/1544)) ([#1556](https://github.com/ooples/AiDotNet/issues/1556)) ([db1c1b9](https://github.com/ooples/AiDotNet/commit/db1c1b9d27930ccee423fb22c8002df03cfefc8e))
* **agentic:** Phase 5 — parity polish (epic [#1544](https://github.com/ooples/AiDotNet/issues/1544)) ([#1557](https://github.com/ooples/AiDotNet/issues/1557)) ([f675cbb](https://github.com/ooples/AiDotNet/commit/f675cbb0ab223e4399f47c9206b6520117143e45))
* **checkpoint:** typed model-state restore via ICheckpointableModel sidecar ([#1811](https://github.com/ooples/AiDotNet/issues/1811)) ([e5395d8](https://github.com/ooples/AiDotNet/commit/e5395d8d4298108de3fff9054ddb21159c2addbc))
* complete all AiDotNet-side JIT items for [#1015](https://github.com/ooples/AiDotNet/issues/1015) (6 of 6) ([#1149](https://github.com/ooples/AiDotNet/issues/1149)) ([fd2c43c](https://github.com/ooples/AiDotNet/commit/fd2c43c336e2d274dcaa3068a8092d31819a9ea1))
* ConfigureTrainingGroups — grouped (per-query-group) training in the facade ([#1580](https://github.com/ooples/AiDotNet/issues/1580)) ([27a7c48](https://github.com/ooples/AiDotNet/commit/27a7c48f40552142f99c09672f630c340bb71c41))
* **credit:** add Local Error Signals + Difference Target Propagation (+ direct variant) credit rules ([#1880](https://github.com/ooples/AiDotNet/issues/1880)) ([1f39e9e](https://github.com/ooples/AiDotNet/commit/1f39e9e8c4b86ee5f375520096c91e234c3e57ba))
* **data:** NCHW layout option for vision data loaders ([e30a47f](https://github.com/ooples/AiDotNet/commit/e30a47ffbf887b8995bb358495732bfdde0bf113))
* **distributed:** ZeRO-Offload equivalent — CPU offload flags on IShardingConfiguration ([#1877](https://github.com/ooples/AiDotNet/issues/1877)) ([2173bc6](https://github.com/ooples/AiDotNet/commit/2173bc6edc09e5636db7b7fb9ccaf6ae3f0bec2d))
* expose GPU diagnostics toggle via AiDotNet builder ([#1122](https://github.com/ooples/AiDotNet/issues/1122)) ([#1147](https://github.com/ooples/AiDotNet/issues/1147)) ([7f9e6d5](https://github.com/ooples/AiDotNet/commit/7f9e6d513ea2215dbe15aa8aaeb2b7e4f2f654c7))
* **facade:** ConfigureTargetScaling — complete the orphaned target-pipeline plumbing ([#1576](https://github.com/ooples/AiDotNet/issues/1576)) ([cdc00a2](https://github.com/ooples/AiDotNet/commit/cdc00a2f30779ece7f6321b1c581074a8628e9ec))
* **facade:** pluggable credit-assignment rules (Feedback Alignment / DFA / Sign-Symmetric) ([#1805](https://github.com/ooples/AiDotNet/issues/1805)) ([e5f25be](https://github.com/ooples/AiDotNet/commit/e5f25be7304c896b90583bc3cfd5ac14171783ed))
* **finance+eval:** options/Kelly/Markowitz/risk-ratios, learning-to-rank, research-eval stats, + 3 ONNX/scaler bug fixes ([#1553](https://github.com/ooples/AiDotNet/issues/1553)) ([6f7e92c](https://github.com/ooples/AiDotNet/commit/6f7e92c4cd9deee3c51481bcf0089415c1aff1d1))
* **finance:** Black-Scholes pricer+Greeks+IV, Kelly sizing, StandardScaler NaN fix ([#1550](https://github.com/ooples/AiDotNet/issues/1550)) ([2cacab8](https://github.com/ooples/AiDotNet/commit/2cacab80de453885d29f94aed6016672c21489c3))
* **finance:** classical volatility model family + options strategy/approval framework ([#1573](https://github.com/ooples/AiDotNet/issues/1573)) ([9f4d66a](https://github.com/ooples/AiDotNet/commit/9f4d66abb9989d97e54c4cea837d84c453586a86))
* **fp16:** extend FP16-activation training to all fused optimizers ([#1543](https://github.com/ooples/AiDotNet/issues/1543)) ([dbd7afb](https://github.com/ooples/AiDotNet/commit/dbd7afbb33fdc43e362c6ab39935fbf7c6db0cc3))
* **fp16:** route SGD compiled training to mixed-dtype plan under AIDOTNET_FP16_ACTIVATIONS ([#1513](https://github.com/ooples/AiDotNet/issues/1513)) ([76060c6](https://github.com/ooples/AiDotNet/commit/76060c6259bdb8e5f698b1cb225e664d4896ea38))
* lazy input-feature ctors for LSTM/GRU/Recurrent/Transformer (closes [#1212](https://github.com/ooples/AiDotNet/issues/1212)) ([#1220](https://github.com/ooples/AiDotNet/issues/1220)) ([f2121da](https://github.com/ooples/AiDotNet/commit/f2121da48a6fd14684cf907c4bcda542453b2f01))
* **licensing:** asymmetric public-key signatures (aidn2) — replace extractable symmetric HMAC ([#1808](https://github.com/ooples/AiDotNet/issues/1808)) ([794c717](https://github.com/ooples/AiDotNet/commit/794c717ad52c53f390f6ad5a79f35a99179d53aa))
* **licensing:** capability-scoped license enforcement (closes [#1195](https://github.com/ooples/AiDotNet/issues/1195)) ([#1196](https://github.com/ooples/AiDotNet/issues/1196)) ([3979389](https://github.com/ooples/AiDotNet/commit/39793894f5b15f924014614818a6e8fcedbfd319))
* **lstm:** wire fused LSTM training path (draft — gated on Tensors [#587](https://github.com/ooples/AiDotNet/issues/587)) ([#1572](https://github.com/ooples/AiDotNet/issues/1572)) ([3119fc8](https://github.com/ooples/AiDotNet/commit/3119fc86169024ce01ef57d04c7be2149de3a0e6))
* **metrics:** language-model perplexity + top-k accuracy metrics ([#1791](https://github.com/ooples/AiDotNet/issues/1791)) ([b09e399](https://github.com/ooples/AiDotNet/commit/b09e399eec3119fddefb18b27d0617d5294f2070))
* **NER:** paper-faithful Word+Char BiLSTM-CRF + fix BiLSTM tape-gradient gaps ([#1636](https://github.com/ooples/AiDotNet/issues/1636)) ([eb259e5](https://github.com/ooples/AiDotNet/commit/eb259e55cac5f219e6895a7b9b46ef6855f8e456))
* **nn:** forward caching allocator — migrate all NeuralNetworkBase to PredictCore, arena default-ON ([#1661](https://github.com/ooples/AiDotNet/issues/1661)) ([#1663](https://github.com/ooples/AiDotNet/issues/1663)) ([7e04670](https://github.com/ooples/AiDotNet/commit/7e0467008074c5516c44926bbed9a51b827a1ebf))
* **nn:** quant-resident inference store selection for foundation-scale models (Phase B, [#1622](https://github.com/ooples/AiDotNet/issues/1622)) ([#1627](https://github.com/ooples/AiDotNet/issues/1627)) ([6588159](https://github.com/ooples/AiDotNet/commit/658815974a011ecbabedd8a08fc9552bf508a390))
* **nn:** recover GPU training faults on the eager CPU path instead of aborting ([#1528](https://github.com/ooples/AiDotNet/issues/1528)) ([81ea753](https://github.com/ooples/AiDotNet/commit/81ea7536dd4220281060cffac043218bab2079fb))
* **onnx:** ONNX export for AiDotNet models + Telco-Churn sample ([#1525](https://github.com/ooples/AiDotNet/issues/1525)) ([27a04d8](https://github.com/ooples/AiDotNet/commit/27a04d834f29c9d340f432a1e05c9791673f8243))
* **perf:** ModelPerfProbe — generic per-model performance + allocation probe ([#1510](https://github.com/ooples/AiDotNet/issues/1510)) ([48dff1a](https://github.com/ooples/AiDotNet/commit/48dff1a3c816ae88d948f192e1a8854268d8d511))
* **rl:** paper-faithful K-step unrolled MuZero training (restore UnrollSteps=5) ([#1759](https://github.com/ooples/AiDotNet/issues/1759)) ([debb40e](https://github.com/ooples/AiDotNet/commit/debb40e4d4413ec5af587c1012903321b1fc5442))
* **streaming:** AiDotNet-side weight streaming for PaLM-E 562B (addresses [#1222](https://github.com/ooples/AiDotNet/issues/1222)) ([#1271](https://github.com/ooples/AiDotNet/issues/1271)) ([e32f180](https://github.com/ooples/AiDotNet/commit/e32f180c0caf8486777d4e277cef93564006f8a4))
* **streaming:** cover every gradient-based optimizer + one-type-per-file split ([#1609](https://github.com/ooples/AiDotNet/issues/1609)) ([9a7c869](https://github.com/ooples/AiDotNet/commit/9a7c86929a015ce423c012c87d048b2e0345e9e3))
* **streaming:** wire automated weight-streaming perf + bump Tensors to 0.95.2 ([#1602](https://github.com/ooples/AiDotNet/issues/1602)) ([7afba76](https://github.com/ooples/AiDotNet/commit/7afba769830f91d6ca9e22626d3bbf9c4d0ae958))
* **timeseries:** DLinear + NLinear + TiDE (current SOTA baselines) + Chronos corpus-scale OOM fix ([#1608](https://github.com/ooples/AiDotNet/issues/1608)) ([9011280](https://github.com/ooples/AiDotNet/commit/90112805e68085428904de69390afe75773aeb30))
* **tools:** DiffusionTraceProbe --unet parallel-efficiency probe ([#642](https://github.com/ooples/AiDotNet/issues/642)) ([#1718](https://github.com/ooples/AiDotNet/issues/1718)) ([bb118e8](https://github.com/ooples/AiDotNet/commit/bb118e8aa7605778cc03daf61a5712c0fb357e0e))
* **training:** GPU-resident fused step for non-TS single-net models ([#1843](https://github.com/ooples/AiDotNet/issues/1843)) ([a5e69ca](https://github.com/ooples/AiDotNet/commit/a5e69cafb03318459389df42eb22267aa8a9a62c))
* **transformer:** opt-in numerically-stable log-softmax-cross-entropy head (default OFF) ([#1828](https://github.com/ooples/AiDotNet/issues/1828)) ([ae6f645](https://github.com/ooples/AiDotNet/commit/ae6f645632acbae5d8e4be572563b17b16f36c78))
* **website:** attribute bulk-issued training licenses by customer email ([#1181](https://github.com/ooples/AiDotNet/issues/1181)) ([9313636](https://github.com/ooples/AiDotNet/commit/931363670c3e4a1bc7e2ebbc99784fd4120c378a))
* **website:** live Stripe Payment Links + post-checkout spinner + webhook provisioning ([#1178](https://github.com/ooples/AiDotNet/issues/1178)) ([cc1a1c0](https://github.com/ooples/AiDotNet/commit/cc1a1c0461bfd1c526b7be9665f2032a377c1c1c))
* **website:** product-aware licensing + Stripe wiring + community-license edge fn ([#1165](https://github.com/ooples/AiDotNet/issues/1165)) ([d06fc21](https://github.com/ooples/AiDotNet/commit/d06fc21b4e48bb7fe58d4282a677dfb523e0a0f6))
* wire JIT compilation through AiModelBuilder ([#1142](https://github.com/ooples/AiDotNet/issues/1142)) ([b006d23](https://github.com/ooples/AiDotNet/commit/b006d231daf8f60e142199eb2bd371e1a5078cd4))


### Bug Fixes

* .clone() the shape array before overwriting the first dim. ([b7e2bf4](https://github.com/ooples/AiDotNet/commit/b7e2bf4d626ce9bd870d5036ea0f4b5ddb85d3a7))
* .clone() the shape array before overwriting the first dim. ([484f295](https://github.com/ooples/AiDotNet/commit/484f295487e4fe993623abb79972897bd90ec010))
* **#1221:** Transformer.Predict bypassed eval-mode wrapper, Dropout fired at inference ([#1242](https://github.com/ooples/AiDotNet/issues/1242)) ([8bc100a](https://github.com/ooples/AiDotNet/commit/8bc100a3155c50cc616a36f623cf5c6423e6316f))
* **#1234, #1235:** SequenceTokenSliceLayer deser branch + horizontal fallback — 258/258 layers + Tensors 0.70.0 + sparse adoption ([#1236](https://github.com/ooples/AiDotNet/issues/1236)) ([5fa2482](https://github.com/ooples/AiDotNet/commit/5fa2482949635350d685e502b11664fd69982fbd))
* **#1238:** Adam8BitOptimizer.Step uses byte[] quantized state — actually delivers 8× memory saving ([#1240](https://github.com/ooples/AiDotNet/issues/1240)) ([cf2dabe](https://github.com/ooples/AiDotNet/commit/cf2dabe79572cf04333a32c3b5c6385520969190))
* **#1245:** ComputeGradients walks GetParameterChunks — also resolves [#1232](https://github.com/ooples/AiDotNet/issues/1232) flat-softmax ([#1247](https://github.com/ooples/AiDotNet/issues/1247)) ([a6e5893](https://github.com/ooples/AiDotNet/commit/a6e589364571a744b91266de310b001905faf7d1))
* **#1296:** chunk full-batch Train + Predict in gradient-based optimizer evaluation path ([#1297](https://github.com/ooples/AiDotNet/issues/1297)) ([8592abb](https://github.com/ooples/AiDotNet/commit/8592abb0218d9c9e5d930b55640b970140aa7206))
* **#1304 c6:** drop Dropout from OccupancyNN defaults; fix memorization invariant ([#1391](https://github.com/ooples/AiDotNet/issues/1391)) ([7207983](https://github.com/ooples/AiDotNet/commit/720798355c20987c65535e69f24558b0a41f7b6c))
* **#1305 cluster-6:** port patchify/unpatchify to FluxDoubleStreamPredictor — fixes 2× output-length shape mismatch ([#1396](https://github.com/ooples/AiDotNet/issues/1396)) ([befe892](https://github.com/ooples/AiDotNet/commit/befe8925ad0cdfefa3d16006adf64978f0dc49e7))
* **#1307:** clusters 4 + 5 — RAPIDFlow/GraFPrint scaffold + RL agent training ([#1316](https://github.com/ooples/AiDotNet/issues/1316)) ([61240b5](https://github.com/ooples/AiDotNet/commit/61240b5690a275a1d445e9f6c31f9a0b0b288a9d))
* **#1307:** dual-precision model-family tests + paper-scale perf ([#1448](https://github.com/ooples/AiDotNet/issues/1448)) ([0db7e68](https://github.com/ooples/AiDotNet/commit/0db7e6805faaeb52bd3aa67a79320ca44bcc2ab0))
* **#1309:** cluster-1 DCGAN — restore deferred-shape guard + lazy-conv deserialize fallback ([#1389](https://github.com/ooples/AiDotNet/issues/1389)) ([ce00cfd](https://github.com/ooples/AiDotNet/commit/ce00cfdd5fb0064928a088280421b569a01113eb))
* **#1311 cluster-3:** snap VLM vision-encoder head count to divide visionDim cleanly ([#1397](https://github.com/ooples/AiDotNet/issues/1397)) ([dcc0aef](https://github.com/ooples/AiDotNet/commit/dcc0aef5001ac74bb6247c7068eeb1c32f97b6c3))
* **#1325:** add InputLayer(int[] outputShape) ctor for InputLayer→MultiHeadAttention chains ([#1326](https://github.com/ooples/AiDotNet/issues/1326)) ([7c6ea0c](https://github.com/ooples/AiDotNet/commit/7c6ea0c4ed47cd55ecf8c964e44d4af34be40489))
* **#1331:** Transformer fused-Adam convergence + sampling/validator/NTM fixes ([#1330](https://github.com/ooples/AiDotNet/issues/1330)) ([48f79fc](https://github.com/ooples/AiDotNet/commit/48f79fc20a38f4da756328f05b96d3c05b160a43))
* **#1332 cluster 4:** deterministic Predict + tape-tracked CRF NLL + Clone weight preservation ([#1356](https://github.com/ooples/AiDotNet/issues/1356)) ([9fe3a19](https://github.com/ooples/AiDotNet/commit/9fe3a1989299858157492e0290733dec153f251f))
* **#1332 cluster 4:** preprocess Train input to MaxSequenceLength in BiLSTMCRF / CNNBiLSTMCRF ([#1339](https://github.com/ooples/AiDotNet/issues/1339)) ([bf77b99](https://github.com/ooples/AiDotNet/commit/bf77b994e0f27a66808f3636cfda346d47eafc6d))
* **#1332 cluster 5:** derive ContinualLearningTestBase.NumParameters from the actual network ([#1337](https://github.com/ooples/AiDotNet/issues/1337)) ([3458cd8](https://github.com/ooples/AiDotNet/commit/3458cd87629ae627705c66b60889c02560fdbdf6))
* **#1340, #1359:** MHA cache lifecycle — clear on eval mode + complete ResetState ([#1366](https://github.com/ooples/AiDotNet/issues/1366)) ([27098ff](https://github.com/ooples/AiDotNet/commit/27098ffef3474796e9f3e62285bd567c0d764fa3))
* **#1349:** SIMD-vectorize INT8 dequant-on-fly matmul in QuantizedDenseLayer + QuantizedAttentionLayer ([#1363](https://github.com/ooples/AiDotNet/issues/1363)) ([158a000](https://github.com/ooples/AiDotNet/commit/158a00003ebb878995881d7b3202e30047c4aa9d))
* **#1354:** wire MixedPrecisionContext through TrainWithTape + expose public API ([#1362](https://github.com/ooples/AiDotNet/issues/1362)) ([7a5a6c8](https://github.com/ooples/AiDotNet/commit/7a5a6c89039145e47eafa5d19041a7d4db27c312))
* **#1355:** auto-record per-thread allocations on ProfilerSession scopes ([#1365](https://github.com/ooples/AiDotNet/issues/1365)) ([4fc3bc1](https://github.com/ooples/AiDotNet/commit/4fc3bc1e1f6330504f135828866716f7f529de34))
* **#1380 + #1382 + #1383:** facade BuildAsync + layers:/ctor validator + consecutive-training determinism ([#1381](https://github.com/ooples/AiDotNet/issues/1381)) ([68f5c69](https://github.com/ooples/AiDotNet/commit/68f5c69acae9de5fc628a8812106eacb52b2f41e))
* **#1380:** set training mode to false for validation/test forward passes in Optimize loop ([#1412](https://github.com/ooples/AiDotNet/issues/1412)) ([679c6c6](https://github.com/ooples/AiDotNet/commit/679c6c644e1e35c421fb374f31459f76f86eebf9))
* **#1380:** transformer residual blocks + audit [#1426](https://github.com/ooples/AiDotNet/issues/1426)/[#1427](https://github.com/ooples/AiDotNet/issues/1427)/[#1428](https://github.com/ooples/AiDotNet/issues/1428) remediation ([#1490](https://github.com/ooples/AiDotNet/issues/1490)) ([d527c23](https://github.com/ooples/AiDotNet/commit/d527c23bdd05d4abf4e3904e20d36bea1d27b019))
* **#1393:** densenet default optimizer adam(1e-3) -&gt; amsgrad-mode adam(1e-4) ([#1403](https://github.com/ooples/AiDotNet/issues/1403)) ([e6ac354](https://github.com/ooples/AiDotNet/commit/e6ac3540c46b6cb99e8cd7b60ac90b685a1dde1f))
* **#1395:** surface caught exception in CompiledTapeTrainingStep fallback ([#1402](https://github.com/ooples/AiDotNet/issues/1402)) ([900134c](https://github.com/ooples/AiDotNet/commit/900134ccf81b832d2ef8ec725ee78bd92b54c0c2))
* **#1400:** swap CrossEntropyLoss → CrossEntropyWithLogitsLoss across 141 files ([#1404](https://github.com/ooples/AiDotNet/issues/1404)) ([7bbfcda](https://github.com/ooples/AiDotNet/commit/7bbfcdac6737624ac6f23fac39aad75ae1cef2f4))
* **#1405:** moe default optimizer overshoots — use amsgrad adam(1e-4) ([#1409](https://github.com/ooples/AiDotNet/issues/1409)) ([2ce0b30](https://github.com/ooples/AiDotNet/commit/2ce0b3027395e1ed7960bb25013b150eee65e315))
* **#1406:** pinn train silently no-op when reusing fused-plan thread cache ([#1411](https://github.com/ooples/AiDotNet/issues/1411)) ([7833258](https://github.com/ooples/AiDotNet/commit/78332587800a820e87718b323fdfd2e2b6ab9ab9))
* **#1407:** rbf scaling-equivariance — deterministic k-means++ center seeding ([#1410](https://github.com/ooples/AiDotNet/issues/1410)) ([159db1b](https://github.com/ooples/AiDotNet/commit/159db1bebbce2cf2e4f9502b7af0a4149a520ab0))
* **#1462:** green the PR [#1455](https://github.com/ooples/AiDotNet/issues/1455) CI shards — diffusion loss, CASTLE/CCM, Siamese clone ([#1467](https://github.com/ooples/AiDotNet/issues/1467)) ([838ed19](https://github.com/ooples/AiDotNet/commit/838ed190433e87098647b1179ce78df5ab650a28))
* **#1468:** AiModelBuilder.BuildAsync on CNN/multi-dim-input NN models ([#1477](https://github.com/ooples/AiDotNet/issues/1477)) ([75d5eef](https://github.com/ooples/AiDotNet/commit/75d5eef7d3e6eb4d56126ff216cfd37f854fc0fc))
* **#1623:** ModelFamily genuine code bugs — paper-faithful fixes across ~25 models ([#1631](https://github.com/ooples/AiDotNet/issues/1631)) ([e04d81c](https://github.com/ooples/AiDotNet/commit/e04d81cf478e530d946d9de115617cbdbf5337f9))
* **#1643:** NTM M-N shard — fused opt-out + deterministic lazy-dense weight resize ([#1687](https://github.com/ooples/AiDotNet/issues/1687)) ([2f48ac5](https://github.com/ooples/AiDotNet/commit/2f48ac54c0392e908d3daaa7ef16d14b0d3cb747))
* **#1668:** enable diffusion denoise-loop inference arena via no_grad-style scope ([#1699](https://github.com/ooples/AiDotNet/issues/1699)) ([595bf31](https://github.com/ooples/AiDotNet/commit/595bf31eb05f118f2378a3c1ebd0d8b2f950dee5))
* **#1670:** training no-op in NeuralTuringMachine, TimeMachine, AudioVisualCorrespondenceNetwork + WhisperTimestamped double-timeout ([#1675](https://github.com/ooples/AiDotNet/issues/1675)) ([dbbd579](https://github.com/ooples/AiDotNet/commit/dbbd579ad87291f654cad3e667f463bbef1954c9))
* **#1675:** green Integration T-Z shard via reproducible init + opt-in LSUV ([#1686](https://github.com/ooples/AiDotNet/issues/1686)) ([31978b0](https://github.com/ooples/AiDotNet/commit/31978b0d57656d06967eadb01cfa4d4037eb403d))
* **#1679/#1624:** run training-perf-bound model-family tests in float, not double ([#1680](https://github.com/ooples/AiDotNet/issues/1680)) ([8ba9fc4](https://github.com/ooples/AiDotNet/commit/8ba9fc4bd7ab5525c7c654055aa45c2a73744ca9))
* **#1713:** two CI timeouts that were hangs/crashes, not heavy — IPW Predict + Meissonic Train ([#1720](https://github.com/ooples/AiDotNet/issues/1720)) ([62841ef](https://github.com/ooples/AiDotNet/commit/62841efd5c01fd5131b62d8f7ce2535b5b24f784))
* 140x optimizer speedup — lazy stats, in-place updates, skip redundant Train() ([#1124](https://github.com/ooples/AiDotNet/issues/1124)) ([a8b316a](https://github.com/ooples/AiDotNet/commit/a8b316a96203aaa5f7e09104c598142c41400fcf))
* 4-model tape-train cascade + HopeNetwork paper-faithful LR ([#1423](https://github.com/ooples/AiDotNet/issues/1423)) ([6ade6c3](https://github.com/ooples/AiDotNet/commit/6ade6c330a0bf8c1a3585ae9b37ba5807b4dec8f))
* **Adam:** AMSGrad optimizer-level fix for cluster 6 post-convergence drift ([#1332](https://github.com/ooples/AiDotNet/issues/1332)) ([#1350](https://github.com/ooples/AiDotNet/issues/1350)) ([433df00](https://github.com/ooples/AiDotNet/commit/433df008cc9412e5e0fc632c4f48a6e92aba1556))
* add a protected generate(shape, steps, seed, initialsample) overload that uses initialsample as the starting noisy sample when supplied (and falls back to fresh gaussian-noise sampling when null). predict copies the user's tensor into an initialsample vector and routes through that overload, so the denoising loop runs starting from the user's input — matching the pytorch diffusers contract `pipeline(latents=user_latents)`. ([4e718fe](https://github.com/ooples/AiDotNet/commit/4e718fec08097a194a8324dde0420661c3778d7e))
* address PR [#1112](https://github.com/ooples/AiDotNet/issues/1112) review round 2 ([60f4120](https://github.com/ooples/AiDotNet/commit/60f41206c851f4c146656c16fb69351565bee566))
* address PR [#1112](https://github.com/ooples/AiDotNet/issues/1112) review round 3 ([066b333](https://github.com/ooples/AiDotNet/commit/066b333843b1a02a54511f9d672819f6606a7d19))
* address PR [#1112](https://github.com/ooples/AiDotNet/issues/1112) review round 3 ([1bc351c](https://github.com/ooples/AiDotNet/commit/1bc351cfb50a8b3bf17c4de2e4fdd745977d7810))
* address PR [#1112](https://github.com/ooples/AiDotNet/issues/1112) review round 4 ([3c4bf10](https://github.com/ooples/AiDotNet/commit/3c4bf10d1fac3f6b117aa32e8a0953fe5b4856f0))
* after fit, return the learned threshold as a single-element vector (subclasses can still override to append additional parameters) ([b7e2bf4](https://github.com/ooples/AiDotNet/commit/b7e2bf4d626ce9bd870d5036ea0f4b5ddb85d3a7))
* after fit, return the learned threshold as a single-element vector (subclasses can still override to append additional parameters) ([484f295](https://github.com/ooples/AiDotNet/commit/484f295487e4fe993623abb79972897bd90ec010))
* **asr:** Conformer/LLM-ASR training — residuals (collapse) + float scaffolds + smoke iterations (double-OOM) ([#1786](https://github.com/ooples/AiDotNet/issues/1786)) ([0a6fc8d](https://github.com/ooples/AiDotNet/commit/0a6fc8d9d4b43a3292e05b985315274a3ed62a24))
* **attention:** resolve lazy shape state before classifying multi-input Forward ([#1585](https://github.com/ooples/AiDotNet/issues/1585)) ([be529fa](https://github.com/ooples/AiDotNet/commit/be529fa437e43a62bdf5ad5749a50c88a822faea))
* **audio:** unblock SenseVoice / Paraformer family — BN→LN + remove broken CIF stub ([#1421](https://github.com/ooples/AiDotNet/issues/1421)) ([98faa97](https://github.com/ooples/AiDotNet/commit/98faa97bc89c9f78e62df54f10e9931dc4f70c74))
* **auth:** surface OAuth errors on /auth/callback instead of silent timeout ([#1258](https://github.com/ooples/AiDotNet/issues/1258)) ([b69c1d1](https://github.com/ooples/AiDotNet/commit/b69c1d103393128f9f835f202aa9de35bbc14b8d))
* **autoencoder:** chain-resolve default layers so EncodedSize is real at construction ([#1587](https://github.com/ooples/AiDotNet/issues/1587)) ([9532c80](https://github.com/ooples/AiDotNet/commit/9532c806a50a1e24c1e0b7296406e82b2afee5d8))
* **batchnorm:** inference broadcast must mirror the channel-axis rule for unbatched rank-3 input ([#1586](https://github.com/ooples/AiDotNet/issues/1586)) ([41abdb5](https://github.com/ooples/AiDotNet/commit/41abdb57d116f40ac9713079e9a3d8a73207f2aa))
* **bench:** pytorch-comparable parity harness — workstation gc, param-matched models, honest rss metric ([#1566](https://github.com/ooples/AiDotNet/issues/1566)) ([#1571](https://github.com/ooples/AiDotNet/issues/1571)) ([61e0fa8](https://github.com/ooples/AiDotNet/commit/61e0fa82167c500b5455d245c63ac35902305061))
* **buildasync-h5:** h5 refuted + restore [#1358](https://github.com/ooples/AiDotNet/issues/1358) fixes + computegradients filter parity with trainwithtape ([#1364](https://github.com/ooples/AiDotNet/issues/1364)) ([ea31261](https://github.com/ooples/AiDotNet/commit/ea3126122af1aa4a7579bce81eb270c086a06572))
* **buildasync:** call registermodel before createmodelversion ([#1367](https://github.com/ooples/AiDotNet/issues/1367)) ([e7c658b](https://github.com/ooples/AiDotNet/commit/e7c658b1d3aaf5fcedd584c370f6a37489739f49))
* **buildkey:** stop ObfuscationTests asserting against the test-override cache ([#1560](https://github.com/ooples/AiDotNet/issues/1560)) ([51de880](https://github.com/ooples/AiDotNet/commit/51de880345364ddbd94eb74c62793c6deb29182b))
* **cgan:** actionable latentDim guard in Predict; correct the noise-size test ([#1588](https://github.com/ooples/AiDotNet/issues/1588)) ([0672eb8](https://github.com/ooples/AiDotNet/commit/0672eb87882ca996eb7a8193bb00b7c4d5340cbf))
* **ci:** cancel old master runs on newer commits + cancel orphaned PR runs on merge ([#1547](https://github.com/ooples/AiDotNet/issues/1547)) ([a517196](https://github.com/ooples/AiDotNet/commit/a5171962cf0c6e82f779784e9f8ba404d32f8301))
* **ci:** close 13 of 20 failing CI shards from PR [#1543](https://github.com/ooples/AiDotNet/issues/1543)'s saved triage ([#1562](https://github.com/ooples/AiDotNet/issues/1562)) ([d9ecc5d](https://github.com/ooples/AiDotNet/commit/d9ecc5d9385d35536216b70192d3021eaa62f65f))
* **ci:** green Diffusion ModelFamily shards — fix DeepFloydIF shape + defer verified foundation-scale OOM models ([#1706](https://github.com/ooples/AiDotNet/issues/1706)) ([#1758](https://github.com/ooples/AiDotNet/issues/1758)) ([3522f76](https://github.com/ooples/AiDotNet/commit/3522f76ff4c3fbe62ba54030da8d6efb2159a3dc))
* **ci:** green ModelFamily NeuralNetworks A-L shard ([#1706](https://github.com/ooples/AiDotNet/issues/1706)) — recurrent-floor tolerances, embedding/VLM HeavyTimeout, DCGAN paper-Adam + GAN invariant, generic streaming-registry reset ([#1742](https://github.com/ooples/AiDotNet/issues/1742)) ([aa56a50](https://github.com/ooples/AiDotNet/commit/aa56a50f10706d241c31fb73eee47a19c9789c99))
* **ci:** green ModelFamily NeuralNetworks T-Z shard ([#1706](https://github.com/ooples/AiDotNet/issues/1706)) ([#1747](https://github.com/ooples/AiDotNet/issues/1747)) ([d345133](https://github.com/ooples/AiDotNet/commit/d3451333b8027512e70885cbf3587fe82a52939d))
* **ci:** green NeuralNetworks A-F + Code/Forecast/Segment/Survival shards (DCGAN/SwinUNETR MoreData timeout) ([#1766](https://github.com/ooples/AiDotNet/issues/1766)) ([39c8242](https://github.com/ooples/AiDotNet/commit/39c824262e3becfb0f02a720a8ec8635efb4b654))
* **ci:** green NeuralNetworks M-N (NTM tolerance) + Unit-10 (MuZero UnrollSteps) shards ([#1755](https://github.com/ooples/AiDotNet/issues/1755)) ([e50bd03](https://github.com/ooples/AiDotNet/commit/e50bd03813c431f8bdde878ca5adfa51ee026b16))
* **ci:** PR [#1563](https://github.com/ooples/AiDotNet/issues/1563) failing-shard lane — diffusion clone, RL/physics ctors, continual-learning + batchnorm tests ([#1565](https://github.com/ooples/AiDotNet/issues/1565)) ([bafda33](https://github.com/ooples/AiDotNet/commit/bafda33826155970a4ccdfce13496270cc5c0c86))
* **ci:** repair 4 master-baseline-broken shards (GP / NN-VLM / 13 / Regression) ([#1461](https://github.com/ooples/AiDotNet/issues/1461)) ([36c74ce](https://github.com/ooples/AiDotNet/commit/36c74ce5587443b69c11ee1d39a23a301af375b1))
* **ci:** repair model invariant failures ([#1597](https://github.com/ooples/AiDotNet/issues/1597)) ([ae4a4bf](https://github.com/ooples/AiDotNet/commit/ae4a4bf1352f4b9985ddeb63d3b2c269920dfa30))
* **ci:** resolve 6 real CI failures + DiT / weight-init vectorization ([#1156](https://github.com/ooples/AiDotNet/issues/1156)) ([15c6f47](https://github.com/ooples/AiDotNet/commit/15c6f47f1790f82406c6f3200c0ae024345b4a7e))
* **ci:** serialize heavy shards to fix runner OOM + reshard diffusion ([#1454](https://github.com/ooples/AiDotNet/issues/1454)) + Tensors 0.91.2 ([#528](https://github.com/ooples/AiDotNet/issues/528)) ([#1485](https://github.com/ooples/AiDotNet/issues/1485)) ([dd40149](https://github.com/ooples/AiDotNet/commit/dd401496ffc40d262a89fc35e61cf6dfe83bcb64))
* **clone:** stop COW layer-walk recursing through pointer fields ([#1669](https://github.com/ooples/AiDotNet/issues/1669)) ([#1676](https://github.com/ooples/AiDotNet/issues/1676)) ([264e75c](https://github.com/ooples/AiDotNet/commit/264e75c94d5ce916de5e0f5dfbc29981e41f1b2d))
* compiled fused-training — standard-Adam default, OCP dispatch, MlpForward wiring, loud fallback ([#1469](https://github.com/ooples/AiDotNet/issues/1469)) ([907bce2](https://github.com/ooples/AiDotNet/commit/907bce2aa32cc725e83fa53182db373958a7364b))
* consolidated AiDotNet fixes + excellence goals + audit pass ([#1832](https://github.com/ooples/AiDotNet/issues/1832), [#1833](https://github.com/ooples/AiDotNet/issues/1833), [#1834](https://github.com/ooples/AiDotNet/issues/1834), [#1835](https://github.com/ooples/AiDotNet/issues/1835), [#1836](https://github.com/ooples/AiDotNet/issues/1836), [#1837](https://github.com/ooples/AiDotNet/issues/1837)) ([#1838](https://github.com/ooples/AiDotNet/issues/1838)) ([1ca524d](https://github.com/ooples/AiDotNet/commit/1ca524d4e224efba9c7b7585818c20d851d7f039))
* convergence-check pattern fix swept across 27 optimizers ([#1351](https://github.com/ooples/AiDotNet/issues/1351) follow-up) ([#1360](https://github.com/ooples/AiDotNet/issues/1360)) ([c70c50a](https://github.com/ooples/AiDotNet/commit/c70c50aba95f4e6cc888b61fd1ee9e5d0f47c378))
* correct sequence layer shape contracts ([#1873](https://github.com/ooples/AiDotNet/issues/1873)) ([f59f3fc](https://github.com/ooples/AiDotNet/commit/f59f3fc6620433aebb1225bebffb2111663b0ae3))
* **cv:** expose ResNet backbone per-stage activations for GetNamedLayerActivations ([#1693](https://github.com/ooples/AiDotNet/issues/1693)) ([5be4b90](https://github.com/ooples/AiDotNet/commit/5be4b9003766a0a47956dd5d79a4e744d746159a))
* **cv:** green CV-segmentation OOM shard — paper-faithful ResNet-50/Swin-L/DCNv3 backbones + in-place Adam ([#1689](https://github.com/ooples/AiDotNet/issues/1689)) ([34ac71c](https://github.com/ooples/AiDotNet/commit/34ac71c49f4d5ce383cf3fbf3971c2f4ddacc0e5))
* **data:** CIFAR/EuroSat NHWC loaders use Tensor&lt;T&gt;.CopyTo (closes [#1151](https://github.com/ooples/AiDotNet/issues/1151)) ([#1154](https://github.com/ooples/AiDotNet/issues/1154)) ([825519c](https://github.com/ooples/AiDotNet/commit/825519ce6d3f79643e64a1154e548f59efdcbeab))
* **data:** repo-wide File.Move/Replace retry via RobustFileOps ([#1153](https://github.com/ooples/AiDotNet/issues/1153)) ([0a33f55](https://github.com/ooples/AiDotNet/commit/0a33f5594feba1d70fdfc24de6422b9a2f9f784f))
* **deps:** bump aidotnet.tensors 0.102.17 to 0.103.1 (streaming releasetopool soft-defer) ([#1695](https://github.com/ooples/AiDotNet/issues/1695)) ([de19dec](https://github.com/ooples/AiDotNet/commit/de19dec5b6afdd9875e1001c6fb3f242d2ca7e26))
* **deps:** bump AiDotNet.Tensors 0.102.9 -&gt; 0.102.12 (conv ArrayPool crash fix) ([#1667](https://github.com/ooples/AiDotNet/issues/1667)) ([01a4ddb](https://github.com/ooples/AiDotNet/commit/01a4ddb4c404ba6d61a8315b5048c3c78ace6b36))
* **deps:** bump AiDotNet.Tensors 0.91.11 → 0.91.12 to unblock master ([#1519](https://github.com/ooples/AiDotNet/issues/1519)) ([97f7567](https://github.com/ooples/AiDotNet/commit/97f75673e916ea43ef136edb92483e346b20444e))
* deserialize weight-preservation ([#1465](https://github.com/ooples/AiDotNet/issues/1465)) + paper-faithful CRF/Donut/NER fixes ([#1466](https://github.com/ooples/AiDotNet/issues/1466)) ([0db695d](https://github.com/ooples/AiDotNet/commit/0db695d1f3b4b6a7b54617d2a04b78d8868be6d0))
* detect latent-shape input in generate, return the latent sample directly with a nan/inf guard but skip the vae decode. pixel-shape inputs still encode/decode as before. ([4e718fe](https://github.com/ooples/AiDotNet/commit/4e718fec08097a194a8324dde0420661c3778d7e))
* detect the multiclass shape ratio up front (predicted.length is an integer multiple of actual.length > 1) and reduce predictions to the true-class probability ([b7e2bf4](https://github.com/ooples/AiDotNet/commit/b7e2bf4d626ce9bd870d5036ea0f4b5ddb85d3a7))
* detect the multiclass shape ratio up front (predicted.length is an integer multiple of actual.length > 1) and reduce predictions to the true-class probability ([484f295](https://github.com/ooples/AiDotNet/commit/484f295487e4fe993623abb79972897bd90ec010))
* **determinism:** seed minibatch shuffle under SetDeterministicMode (real cause of run-to-run training nondeterminism) ([#1819](https://github.com/ooples/AiDotNet/issues/1819)) ([a43234e](https://github.com/ooples/AiDotNet/commit/a43234eb4ccdbadcd611da72d1b3f1ed4ef2e8c3))
* deterministically hash the uuid to a bigint via hashtextextended(text, seed), preserving lock semantics ([683252e](https://github.com/ooples/AiDotNet/commit/683252ebd0503e7b8f441ba2dd2adefb03959fbf))
* **diffusion:** cached posEmbed corrupted by denoise-loop arena reset (non-deterministic Predict, [#1706](https://github.com/ooples/AiDotNet/issues/1706)) ([#1710](https://github.com/ooples/AiDotNet/issues/1710)) ([07dd121](https://github.com/ooples/AiDotNet/commit/07dd121e29157669ca787558c3f82e023fd35697))
* **diffusion:** disable unsafe inference arena in the denoise loop ([#1668](https://github.com/ooples/AiDotNet/issues/1668)) ([#1674](https://github.com/ooples/AiDotNet/issues/1674)) ([7893673](https://github.com/ooples/AiDotNet/commit/78936731f085b0d96ca277a047812280f36f39e5))
* **diffusion:** discover base-class-private predictor params in CollectTrainableParameters walk ([#1707](https://github.com/ooples/AiDotNet/issues/1707)) ([a3799aa](https://github.com/ooples/AiDotNet/commit/a3799aa7ee1cc3a54704cba316d8e34244f4b549))
* **diffusion:** eager per-step denoising forward (avoid compile-cache staleness) + parallel-test BLAS cap ([#1620](https://github.com/ooples/AiDotNet/issues/1620)) ([ac994d6](https://github.com/ooples/AiDotNet/commit/ac994d6d53f19fb7015bcb2b778e0f3778f5d694))
* **diffusion:** green all ModelFamily diffusion shards — clone fix + HeavyTimeout tags + reduced test configs ([#1771](https://github.com/ooples/AiDotNet/issues/1771)) ([e4f6c08](https://github.com/ooples/AiDotNet/commit/e4f6c08835a1a2a05215219dea0322de2a60eada))
* **diffusion:** predictNoiseBatched must not drop the batch dim ([#1843](https://github.com/ooples/AiDotNet/issues/1843) regression) ([#1850](https://github.com/ooples/AiDotNet/issues/1850)) ([d83f043](https://github.com/ooples/AiDotNet/commit/d83f0433340b52f41ff69d2012ac57b3bd623a5a))
* **diffusion:** preserve fp16-resident weights across clone/param round-trip ([#1764](https://github.com/ooples/AiDotNet/issues/1764)) ([#1788](https://github.com/ooples/AiDotNet/issues/1788)) ([c6f0aee](https://github.com/ooples/AiDotNet/commit/c6f0aee5af354589bbb2af08e629b13d0b5f623f))
* **diffusion:** resolve [#1671](https://github.com/ooples/AiDotNet/issues/1671) TCD clone + default-construction timeouts ([#1677](https://github.com/ooples/AiDotNet/issues/1677)) ([8f24939](https://github.com/ooples/AiDotNet/commit/8f24939efb521d84c02711b2a5ece7f29ac5dc11))
* **diffusion:** SASTD 4-ch latent + perf: kill per-forward streaming reflection walk ([#1646](https://github.com/ooples/AiDotNet/issues/1646)) ([#1647](https://github.com/ooples/AiDotNet/issues/1647)) ([b2d307e](https://github.com/ooples/AiDotNet/commit/b2d307e418626449f06a5914e3e40fd8c9c33c40))
* **diffusion:** T5-XXL rent/return weight storage (closes [#1189](https://github.com/ooples/AiDotNet/issues/1189)) ([#1190](https://github.com/ooples/AiDotNet/issues/1190)) ([816d06d](https://github.com/ooples/AiDotNet/commit/816d06db1627ef86043656e86b834ae86e186500))
* **diffusion:** wire predictors into weight streaming ([#1610](https://github.com/ooples/AiDotNet/issues/1610)) ([01d8ad7](https://github.com/ooples/AiDotNet/commit/01d8ad741bbd92b13d37624ac974a5eab844b594))
* **docs:** repair 45 broken links across the GitHub Pages site ([#1522](https://github.com/ooples/AiDotNet/issues/1522)) ([10cebaa](https://github.com/ooples/AiDotNet/commit/10cebaa880fc25939eba964d052a14afaa01d8ea))
* **eigen:** replace o(n^4) single-pivot jacobi with o(n^3) cyclic sweep ([#1231](https://github.com/ooples/AiDotNet/issues/1231)) ([e966541](https://github.com/ooples/AiDotNet/commit/e966541dd941ea7095157e434683dcff95c3711d))
* EmbeddingLayer Optional trainable param ([#1331](https://github.com/ooples/AiDotNet/issues/1331)) + FitDetector rank-discordant leniency ([#1322](https://github.com/ooples/AiDotNet/issues/1322)) ([#1561](https://github.com/ooples/AiDotNet/issues/1561)) ([ddfb03b](https://github.com/ooples/AiDotNet/commit/ddfb03b196e9b083d30a18b920497b2494048e06))
* explicit cast to (iactivationfunction&lt;float&gt;) — the scalar-activation path is what this regression test intends ([ce9318b](https://github.com/ooples/AiDotNet/commit/ce9318bf5808396f4d9827b7b94b228f2093097e))
* explicit SetTrainingMode(false) call right before any of the prediction sub-paths (inference-optimization, jit-compiled, normal model.predict) ([5f6579d](https://github.com/ooples/AiDotNet/commit/5f6579d81d58379083b2de11b0b6fb635cd5f988))
* **facade:** revert unneeded transformer routing; assert REAL learning ([#1803](https://github.com/ooples/AiDotNet/issues/1803)) ([35d55f1](https://github.com/ooples/AiDotNet/commit/35d55f114a29791891aa1b2281356ef36ab9c721))
* **facade:** unblock BuildAsync for radiance-field models ([#1826](https://github.com/ooples/AiDotNet/issues/1826)) ([#1829](https://github.com/ooples/AiDotNet/issues/1829)) ([8909159](https://github.com/ooples/AiDotNet/commit/8909159c60b24239104c64ecdd300652d28aa071))
* **falconmamba:** default to logits-domain cross-entropy so training converges ([#1590](https://github.com/ooples/AiDotNet/issues/1590)) ([b319e0a](https://github.com/ooples/AiDotNet/commit/b319e0a3b0341db506ab7edee5f7ca3dbcf275d9))
* **finance:** clear 8 residual Finance smoke-suite Train/Predict shape drifts ([#1182](https://github.com/ooples/AiDotNet/issues/1182)) ([cdbca71](https://github.com/ooples/AiDotNet/commit/cdbca71ed3b3c473b9ea73a82672bea731d00bff))
* **finance:** defer tsmixer lazy-shape resolution to first forward ([#1712](https://github.com/ooples/AiDotNet/issues/1712)) ([#1716](https://github.com/ooples/AiDotNet/issues/1716)) ([90bc668](https://github.com/ooples/AiDotNet/commit/90bc668604c3df51745226381bcf1e0daa00bf30))
* **finance:** tFT/Informer train through the genuine tape forward (ForwardNativeForTraining) ([#1849](https://github.com/ooples/AiDotNet/issues/1849)) ([47d493e](https://github.com/ooples/AiDotNet/commit/47d493e57d1c7d40ef95ac6f2691670e2aaa89cc))
* **fitness:** R² calculators declared lower-is-better — optimizers kept the WORST iterate ([#1581](https://github.com/ooples/AiDotNet/issues/1581)) ([0b17cf3](https://github.com/ooples/AiDotNet/commit/0b17cf325f51210736b6487efa9f965e9169c63e))
* fix [#1317](https://github.com/ooples/AiDotNet/issues/1317): relax Transformer custom layer validation ([#1320](https://github.com/ooples/AiDotNet/issues/1320)) ([5f4934f](https://github.com/ooples/AiDotNet/commit/5f4934f242f8e0cd9dd808f487d544f6c0507f8d))
* **fp16:** light up the fused FP16 LayerNorm/GELU path on 0.96.0 (StepAdam float args) + e2e test ([#558](https://github.com/ooples/AiDotNet/issues/558)) ([#1604](https://github.com/ooples/AiDotNet/issues/1604)) ([7d2a9d4](https://github.com/ooples/AiDotNet/commit/7d2a9d4d67714a44009d9184db3907b52448fb9d))
* gate the simd path on NETCOREAPP3_0_OR_GREATER — net10.0 production hosts get the vectorized matmul; net471 falls back to the scalar inner loop ([158a000](https://github.com/ooples/AiDotNet/commit/158a00003ebb878995881d7b3202e30047c4aa9d))
* **generators:** exclude non-public types from the YamlTypeRegistry ([#1577](https://github.com/ooples/AiDotNet/issues/1577)) ([e63e193](https://github.com/ooples/AiDotNet/commit/e63e193186535005fd842a2feefb7717b5302b0b))
* **generators:** scope AIDN001 model-metadata validation to the AiDotNet library ([#1825](https://github.com/ooples/AiDotNet/issues/1825)) ([8c6256b](https://github.com/ooples/AiDotNet/commit/8c6256b16a4cd7495fc3fce6264a445350b047d8))
* **gnn:** make NodeClassificationModel actually train ([#1787](https://github.com/ooples/AiDotNet/issues/1787)) ([32f9609](https://github.com/ooples/AiDotNet/commit/32f96094647bd7978a12f194beea0e72e35f8e44))
* **gpu:** don't dispose a forward intermediate aliased by the next layer's reshape view ([#1708](https://github.com/ooples/AiDotNet/issues/1708)) ([f460091](https://github.com/ooples/AiDotNet/commit/f4600914d802cf5c2a0618353bc67ba45c605013))
* **gpu:** invalidate GPU weight cache after the optimizer step (GPU training was frozen) ([#1488](https://github.com/ooples/AiDotNet/issues/1488)) ([137c144](https://github.com/ooples/AiDotNet/commit/137c1449e788e8f5925f7b1d9098ae65ae48a667))
* **gpu:** invalidate resident weight buffers after in-place optimizer update (GPU transformer training was stale) ([#1817](https://github.com/ooples/AiDotNet/issues/1817)) ([5c19829](https://github.com/ooples/AiDotNet/commit/5c1982908ac5f0507fe9b40195c305d4a75d7531))
* **graph:** graph task models require adjacency (strict PyTorch-Geometric contract) ([#1593](https://github.com/ooples/AiDotNet/issues/1593)) ([fe5c38d](https://github.com/ooples/AiDotNet/commit/fe5c38d5efb125f03bbd7c62218f4d12e17e51ae))
* **inference:** flush identity-keyed CPU weight caches after BatchNorm folding ([#1505](https://github.com/ooples/AiDotNet/issues/1505)) ([f610631](https://github.com/ooples/AiDotNet/commit/f610631e7752b28fad3b13b785ba5da9e29c6893))
* **inference:** memory-bounded transformer forward — arena recycles per-layer scratch ([#1824](https://github.com/ooples/AiDotNet/issues/1824)) ([#1824](https://github.com/ooples/AiDotNet/issues/1824)) ([413628d](https://github.com/ooples/AiDotNet/commit/413628d19b366d856892072f1353ce194252769f))
* **init:** seed non-lazy weight init (He/Xavier) from the layer RandomSeed ([#1539](https://github.com/ooples/AiDotNet/issues/1539)) ([1b47680](https://github.com/ooples/AiDotNet/commit/1b47680858390e49adf5a3ce642cf03b7a97b666))
* **layoutxlm:** paper-faithful modality routing + single AdamW step (paper [#3](https://github.com/ooples/AiDotNet/issues/3).1 + [#3](https://github.com/ooples/AiDotNet/issues/3).3) ([#1509](https://github.com/ooples/AiDotNet/issues/1509)) ([f2dcf21](https://github.com/ooples/AiDotNet/commit/f2dcf213c37b00c9b1ec4ffd4c02561826cdc4b7))
* **license, deserialize, tests:** close subclass bypass + fix pipeline interfaces (closes [#1161](https://github.com/ooples/AiDotNet/issues/1161), [#1164](https://github.com/ooples/AiDotNet/issues/1164)) ([#1163](https://github.com/ooples/AiDotNet/issues/1163)) ([c51eaa9](https://github.com/ooples/AiDotNet/commit/c51eaa9768883a23f541775f2768b4654af54891))
* **license:** only classify aidn. keys as offline-HMAC when the signature is 32 bytes ([#1807](https://github.com/ooples/AiDotNet/issues/1807)) ([9e71dea](https://github.com/ooples/AiDotNet/commit/9e71dea6a23a504d2f5a9b14c0078188bda76a56))
* **licenses:** [#1256](https://github.com/ooples/AiDotNet/issues/1256) followup — base64url IsSignedKeyFormat, resend dismiss UX, email branding, e2e row stub ([#1268](https://github.com/ooples/AiDotNet/issues/1268)) ([5124089](https://github.com/ooples/AiDotNet/commit/5124089ebfdcd384badaa796d5b2aa0ffd6e5de2))
* **licenses:** email-on-issuance + admin copy-key & resend-email controls ([#1256](https://github.com/ooples/AiDotNet/issues/1256)) ([683252e](https://github.com/ooples/AiDotNet/commit/683252ebd0503e7b8f441ba2dd2adefb03959fbf))
* **licensing:** building a model must not require a persistence license ([#1574](https://github.com/ooples/AiDotNet/issues/1574)) ([3731b23](https://github.com/ooples/AiDotNet/commit/3731b23890e47363f338caf2b787d385da76aa6c))
* **licensing:** require a prior successful online validation before honouring ValidationPending ([#1802](https://github.com/ooples/AiDotNet/issues/1802)) ([d1afcd0](https://github.com/ooples/AiDotNet/commit/d1afcd0769622e5356f9ee7e8de8797f6f4d2b0c))
* **loss+nn:** re-apply two correctness fixes dropped by the [#1553](https://github.com/ooples/AiDotNet/issues/1553) squash-merge ([#1564](https://github.com/ooples/AiDotNet/issues/1564)) ([ac02ab7](https://github.com/ooples/AiDotNet/commit/ac02ab7242416b94611fe5ee67efc6a8ef79d8c4))
* **loss:** remove double-softmax from CategoricalCrossEntropyLoss.ComputeTapeLoss (closes [#1187](https://github.com/ooples/AiDotNet/issues/1187)) ([#1188](https://github.com/ooples/AiDotNet/issues/1188)) ([b7e2bf4](https://github.com/ooples/AiDotNet/commit/b7e2bf4d626ce9bd870d5036ea0f4b5ddb85d3a7))
* **loss:** sum over class axis in CategoricalCrossEntropyLoss tape (closes [#1191](https://github.com/ooples/AiDotNet/issues/1191)) ([#1192](https://github.com/ooples/AiDotNet/issues/1192)) ([594cae9](https://github.com/ooples/AiDotNet/commit/594cae914df7ed7843b23663cb476e697ddfaee0))
* **lstm:** resolve gate-weight width from the real input, not a stale _inputSize ([#1594](https://github.com/ooples/AiDotNet/issues/1594)) ([68a7c1e](https://github.com/ooples/AiDotNet/commit/68a7c1e93d0c9300dd7a26b324ca6c51f38999ce))
* master CI regressions — clone-bucket serialization, TableTransformer, TimeSeries forecasts ([#1704](https://github.com/ooples/AiDotNet/issues/1704)) ([441c6a9](https://github.com/ooples/AiDotNet/commit/441c6a9d771c87309c397476ae7b724f1fe0222b))
* **meshcnn:** fail fast on empty-mesh (0-edge) input with a clear message ([#1591](https://github.com/ooples/AiDotNet/issues/1591)) ([0d65f65](https://github.com/ooples/AiDotNet/commit/0d65f659c35949b8358a8eeace3da32153f75437))
* **modelfamily:** Generated A-M residual model bugs — BornRule loss + CSPDarknet activations ([#1719](https://github.com/ooples/AiDotNet/issues/1719)) ([d19797b](https://github.com/ooples/AiDotNet/commit/d19797bd2c8466e1ad608fc607848c7e9f4450d3))
* **modelfamily:** paper-faithful 7B-VLA/audio/video model fixes + memory-bounded streaming training ([#1514](https://github.com/ooples/AiDotNet/issues/1514)) ([b414c2f](https://github.com/ooples/AiDotNet/commit/b414c2fde003de15982b0d064ed1d9da86ce1097))
* **modelfamily:** paper-faithful gan 1d-latent rework + serialization symmetry + asr metadata ([#1696](https://github.com/ooples/AiDotNet/issues/1696)) ([eeecf64](https://github.com/ooples/AiDotNet/commit/eeecf6468922010de2770b1d10bad53934f3295d))
* **models:** DCGAN training stability + UnifiedMultimodal streaming collision ([#1737](https://github.com/ooples/AiDotNet/issues/1737), [#1738](https://github.com/ooples/AiDotNet/issues/1738)) ([#1739](https://github.com/ooples/AiDotNet/issues/1739)) ([16c954f](https://github.com/ooples/AiDotNet/commit/16c954ffdcc5a651860c3c8804d7aae8e2bd2884))
* **models:** paper-faithful GAN reshape + Autoformer/Informer convergence (model-bug shards) ([#1606](https://github.com/ooples/AiDotNet/issues/1606)) ([fda2257](https://github.com/ooples/AiDotNet/commit/fda2257cedb876e4abfbefb05f9211619ef728a5))
* **moirai:** unbreak training — tape-safe forward + paper-faithful Adam wiring ([#1516](https://github.com/ooples/AiDotNet/issues/1516)) ([31c29f8](https://github.com/ooples/AiDotNet/commit/31c29f83dfa7e6dc3c882d21a14d1831ca78039a))
* **neural-networks:** lazy-init race in TrainWithTape — warmup before CollectParameters ([#1515](https://github.com/ooples/AiDotNet/issues/1515)) ([c7a6c4b](https://github.com/ooples/AiDotNet/commit/c7a6c4ba93742d3f6eaa22621d8c15bb700fb6d3))
* **neuralnetworks:** align rank-1 regression targets ([B] vs [B,1]) at every training entry point ([#1583](https://github.com/ooples/AiDotNet/issues/1583)) ([27d2f4b](https://github.com/ooples/AiDotNet/commit/27d2f4bc49ccf4a664ae46c57c5afd55457de122))
* **neuralnetworks:** drop using-var disposal of GPU forward result ([#1625](https://github.com/ooples/AiDotNet/issues/1625) / [#1626](https://github.com/ooples/AiDotNet/issues/1626)) ([#1628](https://github.com/ooples/AiDotNet/issues/1628)) ([07db176](https://github.com/ooples/AiDotNet/commit/07db176b31ecff7442b98b80432703e25a0fd251))
* **neuralnetworks:** floor default hidden-layer widths — tiny tabular nets were dead at init ([#1578](https://github.com/ooples/AiDotNet/issues/1578)) ([67a059c](https://github.com/ooples/AiDotNet/commit/67a059c8f6d2e8e2461208e47fd56232af5d2d94))
* **neuralnetworks:** unbreak Transformer gradient flow on tape-based training (closes [#1208](https://github.com/ooples/AiDotNet/issues/1208)) ([#1210](https://github.com/ooples/AiDotNet/issues/1210)) ([dbcb72d](https://github.com/ooples/AiDotNet/commit/dbcb72da94ad5f936eef68e4451797c28f23b503))
* **NN/DNC:** cluster 2 NaN cascade + lazy-tensor materialisation ([#1332](https://github.com/ooples/AiDotNet/issues/1332)) ([#1338](https://github.com/ooples/AiDotNet/issues/1338)) ([84bd106](https://github.com/ooples/AiDotNet/commit/84bd1068172bfbde501ea0383f17eb2eb90338fd))
* **nn/loss:** resolve 27 pre-existing NN/loss integration test failures ([#1652](https://github.com/ooples/AiDotNet/issues/1652)) ([de9a34d](https://github.com/ooples/AiDotNet/commit/de9a34d57a0ebd6bc920b7d2e2f44f2b64677662))
* **NN/NTM:** cluster 1 forward NaN + tape rewrite ([#1332](https://github.com/ooples/AiDotNet/issues/1332)) ([#1335](https://github.com/ooples/AiDotNet/issues/1335)) ([d1e4f76](https://github.com/ooples/AiDotNet/commit/d1e4f761b2b6e2233208fa054ec29f10fa04bc5e))
* **nn:** accept embedding-category custom layers in shape validators ([#1317](https://github.com/ooples/AiDotNet/issues/1317)/[#1321](https://github.com/ooples/AiDotNet/issues/1321)/[#1323](https://github.com/ooples/AiDotNet/issues/1323)) ([#1494](https://github.com/ooples/AiDotNet/issues/1494)) ([ae9c42c](https://github.com/ooples/AiDotNet/commit/ae9c42c4a68a74a9c5fd91296353ba757a04f271))
* **nn:** AudioVisual + CapsuleNetwork Clone/shape/loss fixes (ModelFamily NN) ([#1619](https://github.com/ooples/AiDotNet/issues/1619)) ([68f2e7b](https://github.com/ooples/AiDotNet/commit/68f2e7b240d3d8e54c0cdad1bb0903a3d05e1105))
* **nn:** close 16 pre-existing NN unit-test failures across SSM, embedding, masking, ports, and contracts ([#1424](https://github.com/ooples/AiDotNet/issues/1424)) ([302fb47](https://github.com/ooples/AiDotNet/commit/302fb47751854bda07433e17b225a2750c4efabf))
* **nn:** correct ParameterCount under-report for lazy conv/bn models ([#1688](https://github.com/ooples/AiDotNet/issues/1688)) ([#1692](https://github.com/ooples/AiDotNet/issues/1692)) ([7572c39](https://github.com/ooples/AiDotNet/commit/7572c398f5509e208d96b4fc5cb7efdbddf1307d))
* **nn:** derive weight init from architecture seed (cross-test training determinism) ([#1523](https://github.com/ooples/AiDotNet/issues/1523)) ([7aa209f](https://github.com/ooples/AiDotNet/commit/7aa209f7155d3a7bf4ca848dc292ad0b5e65acbe))
* **nn:** engage weight streaming pre-first-forward for foundation VLMs ([#1621](https://github.com/ooples/AiDotNet/issues/1621)) ([f782c6e](https://github.com/ooples/AiDotNet/commit/f782c6ebf0b6b45724c0024d0165be0489117582))
* **NN:** master CI test failures — SGPT clone, RBM/GraphSAGE gradients, BLAS auto-enable, paper-aligned Word2Vec/Hope ([#1286](https://github.com/ooples/AiDotNet/issues/1286)) ([7712a7b](https://github.com/ooples/AiDotNet/commit/7712a7b51b9deaceb64e858e81024004e04363ff))
* **optimizer:** bound eval cache + add O(tokens) mini-batch-loss fitness mode ([#1820](https://github.com/ooples/AiDotNet/issues/1820)) ([e1e2394](https://github.com/ooples/AiDotNet/commit/e1e239411e29a6855d2a0fa17df316d670b64a7a))
* **optimizer:** bump Tensor.Version after in-place tape Step so GPU re-uploads updated weights ([#1810](https://github.com/ooples/AiDotNet/issues/1810)) ([ca3a1e8](https://github.com/ooples/AiDotNet/commit/ca3a1e8e6057624e804592af7afdcd1cdc67e1d6))
* **optimizers:** bound DefaultGradientCache to stop unbounded training-loop memory leak ([#1831](https://github.com/ooples/AiDotNet/issues/1831)) ([58b6834](https://github.com/ooples/AiDotNet/commit/58b68349ca20a724a2f18ea01b276e4ca8d4bdc0))
* paper-faithful Conv1D + DiffWave/AudioLDM2/ESN/SNN cascade ([#1512](https://github.com/ooples/AiDotNet/issues/1512)) ([75ece09](https://github.com/ooples/AiDotNet/commit/75ece098e393d26dc42b5423d3b71962a9a00e64))
* **post-1219:** wire 0.68.0 capabilities + drive down 359-test CI failure list ([#1225](https://github.com/ooples/AiDotNet/issues/1225)) ([4e718fe](https://github.com/ooples/AiDotNet/commit/4e718fec08097a194a8324dde0420661c3778d7e))
* **PR #1290:** paper-faithful stub implementations + RL agent training + acceleration audit ([#1299](https://github.com/ooples/AiDotNet/issues/1299)) ([a994908](https://github.com/ooples/AiDotNet/commit/a9949089783126cab62d0adda713e6cd63a900ec))
* probe for the 5-arg ctor first (current shape), fall back to the 4-arg ctor for older builds ([c51eaa9](https://github.com/ooples/AiDotNet/commit/c51eaa9768883a23f541775f2768b4654af54891))
* profilersessiontimer now snapshots the per-thread allocation counter on construction and emits the delta to session.recordallocation in stop() ([4fc3bc1](https://github.com/ooples/AiDotNet/commit/4fc3bc1e1f6330504f135828866716f7f529de34))
* promote patchsize to a class-level const near the other fields, deriving patchdim from it and removing the local const in predictnoise ([befe892](https://github.com/ooples/AiDotNet/commit/befe8925ad0cdfefa3d16006adf64978f0dc49e7))
* promote rank-1 → [1, context, 1] and rank-2 → [b, context, 1] at the top of forward, before the embedding layer ([b7e2bf4](https://github.com/ooples/AiDotNet/commit/b7e2bf4d626ce9bd870d5036ea0f4b5ddb85d3a7))
* promote rank-1 → [1, context, 1] and rank-2 → [b, context, 1] at the top of forward, before the embedding layer ([484f295](https://github.com/ooples/AiDotNet/commit/484f295487e4fe993623abb79972897bd90ec010))
* **PTV3:** swap CrossEntropyLoss → CrossEntropyWithLogitsLoss ([#1399](https://github.com/ooples/AiDotNet/issues/1399)) ([b6a80ad](https://github.com/ooples/AiDotNet/commit/b6a80ad545b12edb0812241e25735556e7805f75))
* read ctxlen from features.shape[1] inside the helper so the probe matches the caller's actual width ([6fd6034](https://github.com/ooples/AiDotNet/commit/6fd603481349661ab894db6324d8e795613e7ae6))
* recognise embedding-category layers in custom-chain validators ([#1321](https://github.com/ooples/AiDotNet/issues/1321)/[#1322](https://github.com/ooples/AiDotNet/issues/1322)/[#1323](https://github.com/ooples/AiDotNet/issues/1323)) ([#1324](https://github.com/ooples/AiDotNet/issues/1324)) ([58420a4](https://github.com/ooples/AiDotNet/commit/58420a451b6f6afd5cb0266730fe29bbfc7e3ccf))
* **recurrence:** train xLSTM/GLA/Griffin/Hawk/RecurrentGemma + rebuild LSTMDetector on engine LSTM ([#1595](https://github.com/ooples/AiDotNet/issues/1595)) ([96fa279](https://github.com/ooples/AiDotNet/commit/96fa2791089c4f8ec7f0c00b2cdb53eecaaffae4))
* **release:** wire AIDOTNET_LICENSE_KEY secret into smoke-test gate env ([#1414](https://github.com/ooples/AiDotNet/issues/1414)) ([b35b425](https://github.com/ooples/AiDotNet/commit/b35b425d0cfac6911921dbb06cd006fd4c330982))
* replace the random init + random empty-cluster fallback with a purely deterministic k-means++ farthest-point seeding ([159db1b](https://github.com/ooples/AiDotNet/commit/159db1bebbce2cf2e4f9502b7af0a4149a520ab0)), closes [#1407](https://github.com/ooples/AiDotNet/issues/1407)
* reproduce failing Generated ModelFamily models to match their research papers (VLM family, audio models, DocBank ResNet + framework fixes) ([#1744](https://github.com/ooples/AiDotNet/issues/1744)) ([b552dcd](https://github.com/ooples/AiDotNet/commit/b552dcdde6fd3dee87dc83d671c3c4581096c3e7))
* resolve 11 remaining NN test failures — RBM, Hyperbolic, Hopfield ([#1086](https://github.com/ooples/AiDotNet/issues/1086)) ([af0d7d9](https://github.com/ooples/AiDotNet/commit/af0d7d900d2e2378f2cc7bc6077801ddcca3708b))
* **rl:** DecisionTransformer training, deterministic Predict, and weight-preserving Clone ([#1492](https://github.com/ooples/AiDotNet/issues/1492)) ([a37e201](https://github.com/ooples/AiDotNet/commit/a37e20169276ffc460e5c6720d7be36c98302fc9))
* **rl:** implement real training for CQL and IQL offline agents ([#1728](https://github.com/ooples/AiDotNet/issues/1728)) ([65cdeae](https://github.com/ooples/AiDotNet/commit/65cdeaede65cdaa60822c8c8729bc64182602858))
* **rl:** implement real training for DDPG, TD3, MADDPG, Dreamer, World Models, MuZero ([#1729](https://github.com/ooples/AiDotNet/issues/1729)) ([0dee709](https://github.com/ooples/AiDotNet/commit/0dee709dd9af00f6c153d435dadf27cfb2d95f80))
* route training-path reshape/transpose through tape-safe engine ops ([#1678](https://github.com/ooples/AiDotNet/issues/1678)) ([#1681](https://github.com/ooples/AiDotNet/issues/1681)) ([1f3e39d](https://github.com/ooples/AiDotNet/commit/1f3e39d2e130ab5e965b9ae88cec7f0b1653b83f))
* **samples:** use correct clustering metric property names in customersegmentation ([#1703](https://github.com/ooples/AiDotNet/issues/1703)) ([e38adae](https://github.com/ooples/AiDotNet/commit/e38adaede6c273a5cb4e3377d9914e112dac4254))
* **scaffold:** add Gemma3 + DeepSeekVL/InternVL family to patch-vision list ([#1420](https://github.com/ooples/AiDotNet/issues/1420)) ([390d7ed](https://github.com/ooples/AiDotNet/commit/390d7ede8a575d1ec1b9a0e1d286b276d709eabe))
* **scaffold:** detection backbones — rank-4 InputShape + lazy Conv2D placeholder ParameterCount ([#1517](https://github.com/ooples/AiDotNet/issues/1517)) ([7deb67e](https://github.com/ooples/AiDotNet/commit/7deb67e3d74e9384ae033e0c5776e21b781c41d0))
* **security:** enable RLS on telemetry tables + pin trigger search_path ([#1176](https://github.com/ooples/AiDotNet/issues/1176)) ([075b15b](https://github.com/ooples/AiDotNet/commit/075b15b220bc57de3f3f02c390f49f19125687e3))
* split neuralnetworkbase.serialize / deserialize into a public virtual method (still guarded) plus a private non-virtual helper for deepcopy, so subclass overrides never run during deepcopy ([c51eaa9](https://github.com/ooples/AiDotNet/commit/c51eaa9768883a23f541775f2768b4654af54891))
* **SSM:** rank-1 IOoR in RGLR + restore tape-aware training across 18 LM models ([#1278](https://github.com/ooples/AiDotNet/issues/1278)) ([29fda57](https://github.com/ooples/AiDotNet/commit/29fda5707781fa2fe3456d5abb182c86d4293258))
* **streaming:** wire weight-streaming auto-detect to all Predict paths ([#1520](https://github.com/ooples/AiDotNet/issues/1520)) ([f7bd928](https://github.com/ooples/AiDotNet/commit/f7bd92801865f0870b36b0dac8daad2cc63daeb1))
* **supabase:** drop prior validate_license_key overload + grant ACL on new 5-arg version ([#1215](https://github.com/ooples/AiDotNet/issues/1215)) ([359883b](https://github.com/ooples/AiDotNet/commit/359883b6a641bdfd158f3889714436cea092e7bc))
* **swin:** inference shape bugs (unbatched input + odd-grid padding) — baseline reds ([#1491](https://github.com/ooples/AiDotNet/issues/1491)) ([f1e590c](https://github.com/ooples/AiDotNet/commit/f1e590c8c47ead878d24353077d5ab86b2741334))
* switch the default `_optimizer` from `new adamoptimizer<...>(this)` to `adam(initiallearningrate=1e-4, useamsgrad=true)` ([e6ac354](https://github.com/ooples/AiDotNet/commit/e6ac3540c46b6cb99e8cd7b60ac90b685a1dde1f))
* **synthetic:** fix MedSynth shape bugs (generic-NN reconstruction + constraint broadcast) ([#1507](https://github.com/ooples/AiDotNet/issues/1507)) ([87a7685](https://github.com/ooples/AiDotNet/commit/87a768597608eab3e0641c33fc36a5152b1821af))
* **synthetic:** make AutoDiff-Tab train (real diffusion step) and fix denoiser dims ([#1508](https://github.com/ooples/AiDotNet/issues/1508)) ([d9aceac](https://github.com/ooples/AiDotNet/commit/d9aceac2a41ea2daeab3df21a9c1d17388510e81))
* **synthetic:** make TabTransformerGen trainable with a tape-connected forward ([#1495](https://github.com/ooples/AiDotNet/issues/1495)) ([4004289](https://github.com/ooples/AiDotNet/commit/4004289cbe5417e0f4fb9615a50708fb50496f3b))
* **synthetic:** make TVAE trainable with a tape-connected ELBO step ([#1497](https://github.com/ooples/AiDotNet/issues/1497)) ([2dd914c](https://github.com/ooples/AiDotNet/commit/2dd914c71ede1adee732cca8e1802eb4fd101186))
* **synthetic:** paper-faithful tabular GANs — VGM/copula/sampler + family-wide optimizer-divergence fix (WIP) ([#1589](https://github.com/ooples/AiDotNet/issues/1589)) ([e7c8107](https://github.com/ooples/AiDotNet/commit/e7c8107a7dc6d8c50353b0f1fe582d6fc08b59c7))
* **synthetic:** train FinDiff denoiser via the tape (ε-prediction MSE) ([#1499](https://github.com/ooples/AiDotNet/issues/1499)) ([1604f4d](https://github.com/ooples/AiDotNet/commit/1604f4d2baa8aa217b6e76f4e2a6fedf3882ecb9))
* **synthetic:** train MisGAN via tape-connected WGAN data + mask GANs ([#1502](https://github.com/ooples/AiDotNet/issues/1502)) ([7733c1d](https://github.com/ooples/AiDotNet/commit/7733c1d0ceef6b9cf06e42fe3c4c076b3385cf84))
* **synthetic:** train OCT-GAN via tape-connected SVDD adversarial steps ([#1504](https://github.com/ooples/AiDotNet/issues/1504)) ([2f44657](https://github.com/ooples/AiDotNet/commit/2f4465734dd9895acda5859b51ea70a1435e8207))
* **synthetic:** train PATE-GAN via tape-connected teacher/student/generator steps ([#1503](https://github.com/ooples/AiDotNet/issues/1503)) ([fbc99bd](https://github.com/ooples/AiDotNet/commit/fbc99bd274bcfa001de1be5fba1ea83bd0f2bea0))
* **synthetic:** train TabDDPM denoiser via a tape-connected diffusion loss ([#1498](https://github.com/ooples/AiDotNet/issues/1498)) ([67f4135](https://github.com/ooples/AiDotNet/commit/67f41353eebe3462fb3f279fa9f70d1275cf351b))
* **synthetic:** train TabSyn VAE + latent diffusion via tape-connected steps ([#1500](https://github.com/ooples/AiDotNet/issues/1500)) ([b39b147](https://github.com/ooples/AiDotNet/commit/b39b1478f96abf8ffe2c44e9acb3a5835ea2dbad))
* **tests+graph:** correct 15 NN deep-math tests to actual layer contracts; align LinkPredictionModel adjacency fallback ([#1584](https://github.com/ooples/AiDotNet/issues/1584)) ([788b029](https://github.com/ooples/AiDotNet/commit/788b0299d1c4786749438596d3314fa808412bd8))
* **tests:** net471-portable assertions in MultiHeadAttentionFusedInferenceTests ([#1511](https://github.com/ooples/AiDotNet/issues/1511)) ([eff0527](https://github.com/ooples/AiDotNet/commit/eff0527862999f528e5bf7e1a2f85f77069e945d))
* **timeseries:** make facade training callbacks and early stopping actually work ([#1875](https://github.com/ooples/AiDotNet/issues/1875)) ([0214e23](https://github.com/ooples/AiDotNet/commit/0214e237f90bfef3361730dd382fd5197f6ded1c))
* **timeseries:** Predict() must forecast each row, not return memorized training values ([#1598](https://github.com/ooples/AiDotNet/issues/1598)) ([79948e4](https://github.com/ooples/AiDotNet/commit/79948e48caecea907ec9ad884f99004b4ec7446b))
* track the trainable-layer set's reference identities alongside the cached parameters, forcing invalidation when the layer set changes ([7833258](https://github.com/ooples/AiDotNet/commit/78332587800a820e87718b323fdfd2e2b6ab9ab9)), closes [#1406](https://github.com/ooples/AiDotNet/issues/1406)
* **training:** reclaim fused-optimizer activations per step (bounds [#1624](https://github.com/ooples/AiDotNet/issues/1624)/[#1640](https://github.com/ooples/AiDotNet/issues/1640)) ([#1641](https://github.com/ooples/AiDotNet/issues/1641)) ([02b28fd](https://github.com/ooples/AiDotNet/commit/02b28fda19848f282d80ef817ace795fe5122fa3))
* **training:** Transformer.Train() silent no-op — fused compiled step didn't persist to live params ([#1822](https://github.com/ooples/AiDotNet/issues/1822)) ([#1823](https://github.com/ooples/AiDotNet/issues/1823)) ([803330c](https://github.com/ooples/AiDotNet/commit/803330ca95ae25a3a844cf12f1e3ff81a97bb8d7))
* Transformer training pipeline — feature selection, loss shapes, gradient flatten, download redirect ([#1118](https://github.com/ooples/AiDotNet/issues/1118)) ([1e90fac](https://github.com/ooples/AiDotNet/commit/1e90fac9e9a6e9587f761dffb1b68f2bd10b4819))
* **transformer:** composite-block layout audit — cross-attention, quantization, LoRA, checkpointing all block-aware ([#1493](https://github.com/ooples/AiDotNet/issues/1493)) ([2c7aa51](https://github.com/ooples/AiDotNet/commit/2c7aa51ea4bccfce6996b99794c96f401f69af45))
* **transformer:** default to adam optimizer (vaswani 2017), not vanilla sgd — closes [#1264](https://github.com/ooples/AiDotNet/issues/1264) ([#1265](https://github.com/ooples/AiDotNet/issues/1265)) ([5f6579d](https://github.com/ooples/AiDotNet/commit/5f6579d81d58379083b2de11b0b6fb635cd5f988))
* **transformer:** label smoothing (paper eps=0.1) un-freezes batched training ([#1559](https://github.com/ooples/AiDotNet/issues/1559)) ([#1818](https://github.com/ooples/AiDotNet/issues/1818)) ([f135e7b](https://github.com/ooples/AiDotNet/commit/f135e7b3ae384d7ebf3b9a8497a141a571b22dd8))
* **transformer:** vaswani recipe + working schedule + deterministic init ([#1270](https://github.com/ooples/AiDotNet/issues/1270)) ([e2449b1](https://github.com/ooples/AiDotNet/commit/e2449b1f65fd0c5f610c13003d37719017aeac58))
* **tts:** paper-faithful residual TTS architectures + deterministic inference ([#1527](https://github.com/ooples/AiDotNet/issues/1527)) ([6696fd0](https://github.com/ooples/AiDotNet/commit/6696fd017c1c62c2c61c89066b25e8c5d683869e))
* **vae:** temporalvae decode built decoder temporal layers in the wrong channel order ([#1784](https://github.com/ooples/AiDotNet/issues/1784)) ([e01d3ff](https://github.com/ooples/AiDotNet/commit/e01d3ffa438a12d37cf84606a8fcda4e78457a68))
* **validator:** rank-mismatch + flatten-boundary + custom-layer DeepCopy ([#1333](https://github.com/ooples/AiDotNet/issues/1333)) ([3bd7fc9](https://github.com/ooples/AiDotNet/commit/3bd7fc9d7a3733eaaffde59dbf5575109529a576))
* vectorize the inner loop using portable system.numerics.vector&lt;float&gt;: process contiguous (input, weight) pairs per iteration and accumulate in a vector register ([158a000](https://github.com/ooples/AiDotNet/commit/158a00003ebb878995881d7b3202e30047c4aa9d))
* **vercel:** make website ignore script cwd-independent ([#1542](https://github.com/ooples/AiDotNet/issues/1542)) ([4f6aa2c](https://github.com/ooples/AiDotNet/commit/4f6aa2cd5360f00b934591d9091050310cf6b722))
* **vercel:** stop deploying on every PR — robust change detection in ignoreCommand ([#1554](https://github.com/ooples/AiDotNet/issues/1554)) ([ed2f7cd](https://github.com/ooples/AiDotNet/commit/ed2f7cd7f2fd332681904b45a7d0c7bc275957df))
* **vlm:** correct ViLT fusion architecture + harden MHA head/projection builders ([#1725](https://github.com/ooples/AiDotNet/issues/1725)) ([#1740](https://github.com/ooples/AiDotNet/issues/1740)) ([24d4e68](https://github.com/ooples/AiDotNet/commit/24d4e681aff528b8257c624dc7952d524173d050))
* **vlm:** Q-Former VLMs missing PatchEmbeddingLayer (InstructBLIP+3) ([#1518](https://github.com/ooples/AiDotNet/issues/1518)) ([3003018](https://github.com/ooples/AiDotNet/commit/30030185a8cfa657b370a3db565f94d42fa14dc8))
* **website:** pass Supabase env to Astro build + harden supabase.ts ([#1160](https://github.com/ooples/AiDotNet/issues/1160)) ([c31aeb0](https://github.com/ooples/AiDotNet/commit/c31aeb00d71a37ddda801996d5a858f73f1ba162))
* **website:** payment-flow tier drift, api_usage logging, sign-out clear, error-report mailto ([#1558](https://github.com/ooples/AiDotNet/issues/1558)) ([699804e](https://github.com/ooples/AiDotNet/commit/699804e12249ece940011ab43a57a0ec5ee393be))
* **website:** unblock /auth/callback e2e smoke tests in deploy ([#1282](https://github.com/ooples/AiDotNet/issues/1282)) ([61d338e](https://github.com/ooples/AiDotNet/commit/61d338e6e012c3c8cd03f44bfa1049f44d76c502))
* **website:** unblock production deploy — anchor .vercelignore patterns (aidotnet.dev → 200) ([#1521](https://github.com/ooples/AiDotNet/issues/1521)) ([ff57477](https://github.com/ooples/AiDotNet/commit/ff574779361bb45ea47286c0a34bc418b11d9577))
* **website:** unblock release pipeline — content-layer migration + smoke test fixes ([#1217](https://github.com/ooples/AiDotNet/issues/1217)) ([42d2646](https://github.com/ooples/AiDotNet/commit/42d2646b8c4e2059dabb75d1fd16b58fcd9569e1))
* wire ConfigureAdversarialRobustness through to result + document 4 reserved Configure* methods ([#1357](https://github.com/ooples/AiDotNet/issues/1357) family) ([#1361](https://github.com/ooples/AiDotNet/issues/1361)) ([4be2a69](https://github.com/ooples/AiDotNet/commit/4be2a6969def2fb52409502ddbfbec0deb3b2570))
* wrap each deepcopy's serialize+deserialize pair in modelpersistenceguard.internaloperation(), the existing api savemodel/loadmodel use to suppress the guard ([c51eaa9](https://github.com/ooples/AiDotNet/commit/c51eaa9768883a23f541775f2768b4654af54891))


### Performance

* [codex] add streaming first-order optimizer variants ([#1603](https://github.com/ooples/AiDotNet/issues/1603)) ([4911bc3](https://github.com/ooples/AiDotNet/commit/4911bc3dff9ab37d0971c628d20c04c3f7038729))
* **#1305:** close ConsistencyModel timeout — lazy VAE + 1-step default + semaphore gate ([#1456](https://github.com/ooples/AiDotNet/issues/1456)) ([00e83d7](https://github.com/ooples/AiDotNet/commit/00e83d77297e3be946efaf26ac78bb90d2aa01fd))
* **#1349:** route Int8WeightOnlyMatMul through SgemmWithInt8RowScaledCachedB ([#1417](https://github.com/ooples/AiDotNet/issues/1417)) ([d05f43e](https://github.com/ooples/AiDotNet/commit/d05f43e2dfda5bbc389ba587baebdec0fb6541a1))
* **#1392:** remove O(N²) ordering in NEAT fitness + cache topology sort ([#1419](https://github.com/ooples/AiDotNet/issues/1419)) ([a801d11](https://github.com/ooples/AiDotNet/commit/a801d11c516136df6cdd47a6ef2aa6b483a6beac))
* **#1447:** LSTM fused inference fast path — 198x over per-step loop, beats PyTorch 1.65x ([#1457](https://github.com/ooples/AiDotNet/issues/1457)) ([b24e9e4](https://github.com/ooples/AiDotNet/commit/b24e9e429a861b9c373f50a82cc8fe51a3c81f19))
* **#1458:** fast-path single-call Vector SafetyFilter input/output paths ([#1475](https://github.com/ooples/AiDotNet/issues/1475)) ([2ede28c](https://github.com/ooples/AiDotNet/commit/2ede28cefe1291e30cf5e6f1c764ca1beb75741f))
* **#1464:** fix MGTSD + RWKV7Block training-throughput timeouts ([#1471](https://github.com/ooples/AiDotNet/issues/1471)) ([70427f9](https://github.com/ooples/AiDotNet/commit/70427f9122db582c075b9f62e9d67fc1c44c9450))
* **#1478:** route Transformer self-attention inference to fused MHA kernel (P0) ([#1489](https://github.com/ooples/AiDotNet/issues/1489)) ([eecd7cb](https://github.com/ooples/AiDotNet/commit/eecd7cba9f92c42f569daa47520e0516a78e533b))
* **#1624:** optimizer ladder + COW clone (fixed) + streaming param setter + lazy foundation-scale construct ([#1633](https://github.com/ooples/AiDotNet/issues/1633)) ([c331847](https://github.com/ooples/AiDotNet/commit/c331847d5d3b3ff55c285985ae8166c8df933a16))
* **#1662:** bit-identical fused optimizer-in-backward (lever [#1](https://github.com/ooples/AiDotNet/issues/1)) ([#1664](https://github.com/ooples/AiDotNet/issues/1664)) ([972a8eb](https://github.com/ooples/AiDotNet/commit/972a8eb0e2ae61908f0633aae7e89bc50a538870))
* **#1672:** fp16-resident diffusion weights + faithful-DiT AdaLN fix ([#1682](https://github.com/ooples/AiDotNet/issues/1682)) ([d154227](https://github.com/ooples/AiDotNet/commit/d154227eda8bedc9830ec52cbffb8575b1004e74))
* **#1672:** gated *Into resident-scratch + fused-QKV for DiT/SiT diffusion inference ([#1697](https://github.com/ooples/AiDotNet/issues/1697)) ([87bd969](https://github.com/ooples/AiDotNet/commit/87bd9692f7fbb9fb385ecc830bbe88c33acb836f))
* **#653:** reuse one TensorArena across grad-accumulation chunks (caching allocator) ([#1651](https://github.com/ooples/AiDotNet/issues/1651)) ([d079565](https://github.com/ooples/AiDotNet/commit/d079565e6e52a56cb9c1708fbec187b23f73db55))
* activate TensorArena for training across all model paths (zero-alloc) ([#1809](https://github.com/ooples/AiDotNet/issues/1809)) ([a39fb1b](https://github.com/ooples/AiDotNet/commit/a39fb1bae59b737904696e94b03ff2452c06c4e6))
* cache Adam per-param backing arrays across steps (+ NBEATS bias-col reuse) ([#1816](https://github.com/ooples/AiDotNet/issues/1816)) ([29d2ad4](https://github.com/ooples/AiDotNet/commit/29d2ad468a6c30bb49764a48f767f00b6d4148b7))
* **cholesky:** offset-0 copy-buffer dot — KernelRidge/GP 42GB-&gt;1.6GB (Tensors [#575](https://github.com/ooples/AiDotNet/issues/575) shipped in 0.92.0) ([#1546](https://github.com/ooples/AiDotNet/issues/1546)) ([af7b5d4](https://github.com/ooples/AiDotNet/commit/af7b5d4ebadacc45427e5d7d58ac8c2daf4cb00d))
* **densenet:** replace O(n²) IndexOf-in-foreach with index loop in DenseBlock.SetExtraParameters ([#1241](https://github.com/ooples/AiDotNet/issues/1241)) ([7785696](https://github.com/ooples/AiDotNet/commit/77856963a67b4a51baa5d8ee6f1cb2d17ef7ed5d))
* deterministic-by-default on AiModelBuilder + .AllowNondeterminism() opt-out ([#1145](https://github.com/ooples/AiDotNet/issues/1145)) ([f2acfe7](https://github.com/ooples/AiDotNet/commit/f2acfe7c163966a27dc4333452eae8b8b3480631))
* **diffusion:** paper-faithful Adam optimizer in the shared training step ([#1748](https://github.com/ooples/AiDotNet/issues/1748)) ([aeb1f57](https://github.com/ooples/AiDotNet/commit/aeb1f5767efb916a8d3979fd793d33192a9a7b26))
* fix timing-out Integration A-B/C/D shards (AnoGAN analytic grad + parallel streaming Adam) ([#1599](https://github.com/ooples/AiDotNet/issues/1599)) ([c05d8f4](https://github.com/ooples/AiDotNet/commit/c05d8f4ab256373bf471cd93de7bd497f9d36f24))
* force loh-compacting gc between unit-03 diffusion tests ([#1136](https://github.com/ooples/AiDotNet/issues/1136)) ([#1148](https://github.com/ooples/AiDotNet/issues/1148)) ([9cd5416](https://github.com/ooples/AiDotNet/commit/9cd54164212274257bed5cd7f96469532e1691a5))
* fused fwd+bwd+optimizer as one compiled kernel in TrainWithTape ([#1144](https://github.com/ooples/AiDotNet/issues/1144)) ([dda85bb](https://github.com/ooples/AiDotNet/commit/dda85bb2b74cd6255b58d7d118840844df6830cc))
* **gnn:** raw-array GraphAttentionLayer dense scores + backward (kill per-access index arrays) ([#1701](https://github.com/ooples/AiDotNet/issues/1701)) ([e0df6af](https://github.com/ooples/AiDotNet/commit/e0df6af560381b74f1a98ffd9c83f48d606bbe0c))
* **gp:** fix GPWithMCMC 60s timeout in Clustering/GP shard (cache invariant base kernel) ([#1763](https://github.com/ooples/AiDotNet/issues/1763)) ([f108ba1](https://github.com/ooples/AiDotNet/commit/f108ba14ffd74c0649059de56adce0f5a3446ad7))
* **gpu:** GPU-resident optimizer step (Adam validated; AdamW/SGD to follow) - host-read-free step for cudaGraph ([#1501](https://github.com/ooples/AiDotNet/issues/1501)) ([8603fbe](https://github.com/ooples/AiDotNet/commit/8603fbe340fd82884471d260df02c6c78a3521c0))
* **gpu:** make parameters GPU-resident so the compiled-plan GPU Adam fires ([#1601](https://github.com/ooples/AiDotNet/issues/1601)) ([fd8a882](https://github.com/ooples/AiDotNet/commit/fd8a882057f7f15b5a1144fb27c1ea683d9428f7))
* **inference:** freeze-time BatchNorm folding (Conv/Dense → BN) in InferenceOptimizer (Phase 6) ([#1473](https://github.com/ooples/AiDotNet/issues/1473)) ([b5916db](https://github.com/ooples/AiDotNet/commit/b5916dbd4a8af7331237addd2dfcd775b55c492f))
* **layers:** lazy weight init in video VLM helper — partial [#1136](https://github.com/ooples/AiDotNet/issues/1136) ([#1194](https://github.com/ooples/AiDotNet/issues/1194)) ([77727a4](https://github.com/ooples/AiDotNet/commit/77727a4bdca7f04c5f9aba782109ea9c8248db43))
* **layers:** skip dead manual-backward activation caches under tape (minor; not the [#1624](https://github.com/ooples/AiDotNet/issues/1624) fix) ([#1637](https://github.com/ooples/AiDotNet/issues/1637)) ([f226767](https://github.com/ooples/AiDotNet/commit/f226767269f8b87c6a7f4edab51f859be03558bf))
* lazy attention + Dispose-to-pool cascade (parts 2+3 of [#1136](https://github.com/ooples/AiDotNet/issues/1136)) ([#1140](https://github.com/ooples/AiDotNet/issues/1140)) ([3ffe067](https://github.com/ooples/AiDotNet/commit/3ffe06774e108b77d32befe2b7da1667f7977476))
* lazy init for VGG + CapsuleNetwork to unblock NeuralNetworks ModelFamily shard ([#1138](https://github.com/ooples/AiDotNet/issues/1138)) ([0a12ca1](https://github.com/ooples/AiDotNet/commit/0a12ca1790e3ebaa610d7ba6d895f399b21e80f2))
* lazy init in diffusion noise predictors + GC between tests ([#1137](https://github.com/ooples/AiDotNet/issues/1137)) ([bd0a1c3](https://github.com/ooples/AiDotNet/commit/bd0a1c3ce2c0d6e4a12eec5c1e0c991b274cdfbf))
* **optimizer:** parallelize eager Adam Step across parameters ([#1806](https://github.com/ooples/AiDotNet/issues/1806)) ([0b3adac](https://github.com/ooples/AiDotNet/commit/0b3adac800f0804534519eec16be7f08d0deb213))
* **optimizer:** raw-array Adam8Bit BF16 quant/dequant (PerfView NN-shard hot path) ([#1698](https://github.com/ooples/AiDotNet/issues/1698)) ([5cc72a5](https://github.com/ooples/AiDotNet/commit/5cc72a5d4a099b03713044071cb04ffe283f1e61))
* **optimizers:** sparse-by-default — all 19 dense paths consume sparse via ToDense, Adam/AdamW scatter ([#1526](https://github.com/ooples/AiDotNet/issues/1526)) ([2dfb5d0](https://github.com/ooples/AiDotNet/commit/2dfb5d0787f17b084abe865cd6c391ac1d6e5698))
* **optimizers:** wire amsgrad to the fused compiled training path ([#1653](https://github.com/ooples/AiDotNet/issues/1653)) ([062aa37](https://github.com/ooples/AiDotNet/commit/062aa37fdccf84c387b531a581eff5f14339da70))
* **optimizer:** wire eager fp32 Adam step onto the shared SIMD kernel ([#1815](https://github.com/ooples/AiDotNet/issues/1815)) ([1a5125c](https://github.com/ooples/AiDotNet/commit/1a5125ca9e73f9a57e96ac9777efa5e51ff67235))
* perf+fix([#1464](https://github.com/ooples/AiDotNet/issues/1464)): vectorize RWKVLayer + make it differentiable (WIP) ([#1472](https://github.com/ooples/AiDotNet/issues/1472)) ([805f1ba](https://github.com/ooples/AiDotNet/commit/805f1ba38b7e0b03ee8eaa7e152c8e8bfb4a2901))
* **regression:** cut tree-model training allocation (zero per-node/per-threshold copies) ([#1531](https://github.com/ooples/AiDotNet/issues/1531)) ([b404df9](https://github.com/ooples/AiDotNet/commit/b404df97b079680e6fcb1525c42772d106181228))
* **rl:** batch trading-agent updates — 16x faster training ([#1529](https://github.com/ooples/AiDotNet/issues/1529)) ([103fd52](https://github.com/ooples/AiDotNet/commit/103fd52f081ecda9fd603a35836355118405be48))
* scalar loops → Engine ops in hot-path layers (part 4 of [#1136](https://github.com/ooples/AiDotNet/issues/1136)) ([#1141](https://github.com/ooples/AiDotNet/issues/1141)) ([7c65bea](https://github.com/ooples/AiDotNet/commit/7c65bea8a6024558b719a21ab9050cfd6f1afd6c))
* SimdRandom + lazy init + bulk copy — fix model test timeouts ([#1133](https://github.com/ooples/AiDotNet/issues/1133)) ([72f425a](https://github.com/ooples/AiDotNet/commit/72f425a84337a9a7ff85f704a248fb00eaabccbe))
* **training:** make BF16-Adam fused-compatible (proper bf16 moment kernel, not a gate) ([#1745](https://github.com/ooples/AiDotNet/issues/1745)) ([fbe7fe3](https://github.com/ooples/AiDotNet/commit/fbe7fe37695c1a6a2ea149acdce598f7fdcacdeb))
* use Engine.TensorPermute + Span.CopyTo for layout conversion ([dea8bc6](https://github.com/ooples/AiDotNet/commit/dea8bc66406301c724bffba6e6b54045a85a0084))
* wire gradient checkpointing into ForwardForTraining via existing TrainingMemoryConfig ([#1146](https://github.com/ooples/AiDotNet/issues/1146)) ([ac323a9](https://github.com/ooples/AiDotNet/commit/ac323a947659da945199b6e064dbc5060033cd08))


### Reverts

* drop the system.numerics.vector path entirely — scalar inner loop remains correct (just slow); proper SIMD speedup belongs in tensors via SgemmWithInt8CachedB ([158a000](https://github.com/ooples/AiDotNet/commit/158a00003ebb878995881d7b3202e30047c4aa9d))


### Refactoring

* CompiledModelHost foundation + Dispose cascade across NN/Diffusion ([#1143](https://github.com/ooples/AiDotNet/issues/1143)) ([ebad581](https://github.com/ooples/AiDotNet/commit/ebad5810f88d6b058b6b614f0bcd0f2badf3529c))
* **conditioner:** introduce compositeconditioningbase for engine access ([#1233](https://github.com/ooples/AiDotNet/issues/1233)) ([f88c240](https://github.com/ooples/AiDotNet/commit/f88c240bcdc67858d9450cbc8038e91828aa4d22))


### Build System

* 0 errors net10.0 (main + testconsole + tests). ([cdbca71](https://github.com/ooples/AiDotNet/commit/cdbca71ed3b3c473b9ea73a82672bea731d00bff))
* 0 errors net10.0. ([cdbca71](https://github.com/ooples/AiDotNet/commit/cdbca71ed3b3c473b9ea73a82672bea731d00bff))
* 0 errors net10.0. ([cdbca71](https://github.com/ooples/AiDotNet/commit/cdbca71ed3b3c473b9ea73a82672bea731d00bff))
* 0 errors net10.0. all 3 + 4 integration tests pass. ([b7e2bf4](https://github.com/ooples/AiDotNet/commit/b7e2bf4d626ce9bd870d5036ea0f4b5ddb85d3a7))
* 0 errors net10.0. all 3 + 4 integration tests pass. ([484f295](https://github.com/ooples/AiDotNet/commit/484f295487e4fe993623abb79972897bd90ec010))
* bump AiDotNet.Tensors to 0.84.1 ([#1460](https://github.com/ooples/AiDotNet/issues/1460)) ([55f4ecd](https://github.com/ooples/AiDotNet/commit/55f4ecd2bd02121321001a30f486dfc10bb143bd))
* **deps:** bump aidotnet.tensors + native packages to 0.102.9 ([#1665](https://github.com/ooples/AiDotNet/issues/1665)) ([06b470e](https://github.com/ooples/AiDotNet/commit/06b470ea274f7f9df7eb1935e563d7a056363030))
* **deps:** bump astro ([#1648](https://github.com/ooples/AiDotNet/issues/1648)) ([c8b6397](https://github.com/ooples/AiDotNet/commit/c8b63979cbc456f1a9c999025803152147b0f8c7))
* **deps:** bump the npm_and_yarn group across 1 directory with 2 updates ([#1660](https://github.com/ooples/AiDotNet/issues/1660)) ([a31cd25](https://github.com/ooples/AiDotNet/commit/a31cd25abb23f5325e8f031b42767f258a453565))
* **deps:** bump the npm_and_yarn group across 2 directories with 3 updates ([#1649](https://github.com/ooples/AiDotNet/issues/1649)) ([7a46620](https://github.com/ooples/AiDotNet/commit/7a466205eab5149ecb4a14152b3ef88f7f320a30))
* net10.0 success, net471 falls into the -1L branch (compile-time gated via NETCOREAPP3_0_OR_GREATER) ([4fc3bc1](https://github.com/ooples/AiDotNet/commit/4fc3bc1e1f6330504f135828866716f7f529de34))
* ship XML documentation in NuGet packages ([#1644](https://github.com/ooples/AiDotNet/issues/1644)) ([f7d45d0](https://github.com/ooples/AiDotNet/commit/f7d45d0b431e20f75fa9fc95c104819a8231a3cb))


### Documentation

* audit(1425-1428): document DRM, extract opt-in metapackages, learnable VLA generation modules ([#1487](https://github.com/ooples/AiDotNet/issues/1487)) ([5107a83](https://github.com/ooples/AiDotNet/commit/5107a83b55b4fdc6aad77c3866b41d88f31b7e9c))
* **serving:** startup CPU-inference BLAS thread-pin recommendation ([#1474](https://github.com/ooples/AiDotNet/issues/1474)) ([3af86c9](https://github.com/ooples/AiDotNet/commit/3af86c99cd2bdc2c956daa65e0bbb8c1924fb7a6))

## [0.230.0](https://github.com/ooples/AiDotNet/compare/v0.229.2...v0.230.0) (2026-07-17)


### Features

* **checkpoint:** typed model-state restore via ICheckpointableModel sidecar ([#1811](https://github.com/ooples/AiDotNet/issues/1811)) ([e5395d8](https://github.com/ooples/AiDotNet/commit/e5395d8d4298108de3fff9054ddb21159c2addbc))
* **credit:** add Local Error Signals + Difference Target Propagation (+ direct variant) credit rules ([#1880](https://github.com/ooples/AiDotNet/issues/1880)) ([1f39e9e](https://github.com/ooples/AiDotNet/commit/1f39e9e8c4b86ee5f375520096c91e234c3e57ba))
* **distributed:** ZeRO-Offload equivalent — CPU offload flags on IShardingConfiguration ([#1877](https://github.com/ooples/AiDotNet/issues/1877)) ([2173bc6](https://github.com/ooples/AiDotNet/commit/2173bc6edc09e5636db7b7fb9ccaf6ae3f0bec2d))
* **facade:** pluggable credit-assignment rules (Feedback Alignment / DFA / Sign-Symmetric) ([#1805](https://github.com/ooples/AiDotNet/issues/1805)) ([e5f25be](https://github.com/ooples/AiDotNet/commit/e5f25be7304c896b90583bc3cfd5ac14171783ed))
* **licensing:** asymmetric public-key signatures (aidn2) — replace extractable symmetric HMAC ([#1808](https://github.com/ooples/AiDotNet/issues/1808)) ([794c717](https://github.com/ooples/AiDotNet/commit/794c717ad52c53f390f6ad5a79f35a99179d53aa))
* **metrics:** language-model perplexity + top-k accuracy metrics ([#1791](https://github.com/ooples/AiDotNet/issues/1791)) ([b09e399](https://github.com/ooples/AiDotNet/commit/b09e399eec3119fddefb18b27d0617d5294f2070))
* **training:** GPU-resident fused step for non-TS single-net models ([#1843](https://github.com/ooples/AiDotNet/issues/1843)) ([a5e69ca](https://github.com/ooples/AiDotNet/commit/a5e69cafb03318459389df42eb22267aa8a9a62c))
* **transformer:** opt-in numerically-stable log-softmax-cross-entropy head (default OFF) ([#1828](https://github.com/ooples/AiDotNet/issues/1828)) ([ae6f645](https://github.com/ooples/AiDotNet/commit/ae6f645632acbae5d8e4be572563b17b16f36c78))


### Bug Fixes

* **ci:** green Diffusion ModelFamily shards — fix DeepFloydIF shape + defer verified foundation-scale OOM models ([#1706](https://github.com/ooples/AiDotNet/issues/1706)) ([#1758](https://github.com/ooples/AiDotNet/issues/1758)) ([3522f76](https://github.com/ooples/AiDotNet/commit/3522f76ff4c3fbe62ba54030da8d6efb2159a3dc))
* consolidated AiDotNet fixes + excellence goals + audit pass ([#1832](https://github.com/ooples/AiDotNet/issues/1832), [#1833](https://github.com/ooples/AiDotNet/issues/1833), [#1834](https://github.com/ooples/AiDotNet/issues/1834), [#1835](https://github.com/ooples/AiDotNet/issues/1835), [#1836](https://github.com/ooples/AiDotNet/issues/1836), [#1837](https://github.com/ooples/AiDotNet/issues/1837)) ([#1838](https://github.com/ooples/AiDotNet/issues/1838)) ([1ca524d](https://github.com/ooples/AiDotNet/commit/1ca524d4e224efba9c7b7585818c20d851d7f039))
* correct sequence layer shape contracts ([#1873](https://github.com/ooples/AiDotNet/issues/1873)) ([f59f3fc](https://github.com/ooples/AiDotNet/commit/f59f3fc6620433aebb1225bebffb2111663b0ae3))
* **determinism:** seed minibatch shuffle under SetDeterministicMode (real cause of run-to-run training nondeterminism) ([#1819](https://github.com/ooples/AiDotNet/issues/1819)) ([a43234e](https://github.com/ooples/AiDotNet/commit/a43234eb4ccdbadcd611da72d1b3f1ed4ef2e8c3))
* **diffusion:** predictNoiseBatched must not drop the batch dim ([#1843](https://github.com/ooples/AiDotNet/issues/1843) regression) ([#1850](https://github.com/ooples/AiDotNet/issues/1850)) ([d83f043](https://github.com/ooples/AiDotNet/commit/d83f0433340b52f41ff69d2012ac57b3bd623a5a))
* **diffusion:** preserve fp16-resident weights across clone/param round-trip ([#1764](https://github.com/ooples/AiDotNet/issues/1764)) ([#1788](https://github.com/ooples/AiDotNet/issues/1788)) ([c6f0aee](https://github.com/ooples/AiDotNet/commit/c6f0aee5af354589bbb2af08e629b13d0b5f623f))
* **facade:** revert unneeded transformer routing; assert REAL learning ([#1803](https://github.com/ooples/AiDotNet/issues/1803)) ([35d55f1](https://github.com/ooples/AiDotNet/commit/35d55f114a29791891aa1b2281356ef36ab9c721))
* **facade:** unblock BuildAsync for radiance-field models ([#1826](https://github.com/ooples/AiDotNet/issues/1826)) ([#1829](https://github.com/ooples/AiDotNet/issues/1829)) ([8909159](https://github.com/ooples/AiDotNet/commit/8909159c60b24239104c64ecdd300652d28aa071))
* **finance:** tFT/Informer train through the genuine tape forward (ForwardNativeForTraining) ([#1849](https://github.com/ooples/AiDotNet/issues/1849)) ([47d493e](https://github.com/ooples/AiDotNet/commit/47d493e57d1c7d40ef95ac6f2691670e2aaa89cc))
* **generators:** scope AIDN001 model-metadata validation to the AiDotNet library ([#1825](https://github.com/ooples/AiDotNet/issues/1825)) ([8c6256b](https://github.com/ooples/AiDotNet/commit/8c6256b16a4cd7495fc3fce6264a445350b047d8))
* **gpu:** invalidate resident weight buffers after in-place optimizer update (GPU transformer training was stale) ([#1817](https://github.com/ooples/AiDotNet/issues/1817)) ([5c19829](https://github.com/ooples/AiDotNet/commit/5c1982908ac5f0507fe9b40195c305d4a75d7531))
* **inference:** memory-bounded transformer forward — arena recycles per-layer scratch ([#1824](https://github.com/ooples/AiDotNet/issues/1824)) ([#1824](https://github.com/ooples/AiDotNet/issues/1824)) ([413628d](https://github.com/ooples/AiDotNet/commit/413628d19b366d856892072f1353ce194252769f))
* **license:** only classify aidn. keys as offline-HMAC when the signature is 32 bytes ([#1807](https://github.com/ooples/AiDotNet/issues/1807)) ([9e71dea](https://github.com/ooples/AiDotNet/commit/9e71dea6a23a504d2f5a9b14c0078188bda76a56))
* **licensing:** require a prior successful online validation before honouring ValidationPending ([#1802](https://github.com/ooples/AiDotNet/issues/1802)) ([d1afcd0](https://github.com/ooples/AiDotNet/commit/d1afcd0769622e5356f9ee7e8de8797f6f4d2b0c))
* **optimizer:** bound eval cache + add O(tokens) mini-batch-loss fitness mode ([#1820](https://github.com/ooples/AiDotNet/issues/1820)) ([e1e2394](https://github.com/ooples/AiDotNet/commit/e1e239411e29a6855d2a0fa17df316d670b64a7a))
* **optimizer:** bump Tensor.Version after in-place tape Step so GPU re-uploads updated weights ([#1810](https://github.com/ooples/AiDotNet/issues/1810)) ([ca3a1e8](https://github.com/ooples/AiDotNet/commit/ca3a1e8e6057624e804592af7afdcd1cdc67e1d6))
* **optimizers:** bound DefaultGradientCache to stop unbounded training-loop memory leak ([#1831](https://github.com/ooples/AiDotNet/issues/1831)) ([58b6834](https://github.com/ooples/AiDotNet/commit/58b68349ca20a724a2f18ea01b276e4ca8d4bdc0))
* **timeseries:** make facade training callbacks and early stopping actually work ([#1875](https://github.com/ooples/AiDotNet/issues/1875)) ([0214e23](https://github.com/ooples/AiDotNet/commit/0214e237f90bfef3361730dd382fd5197f6ded1c))
* **training:** Transformer.Train() silent no-op — fused compiled step didn't persist to live params ([#1822](https://github.com/ooples/AiDotNet/issues/1822)) ([#1823](https://github.com/ooples/AiDotNet/issues/1823)) ([803330c](https://github.com/ooples/AiDotNet/commit/803330ca95ae25a3a844cf12f1e3ff81a97bb8d7))
* **transformer:** label smoothing (paper eps=0.1) un-freezes batched training ([#1559](https://github.com/ooples/AiDotNet/issues/1559)) ([#1818](https://github.com/ooples/AiDotNet/issues/1818)) ([f135e7b](https://github.com/ooples/AiDotNet/commit/f135e7b3ae384d7ebf3b9a8497a141a571b22dd8))


### Performance

* activate TensorArena for training across all model paths (zero-alloc) ([#1809](https://github.com/ooples/AiDotNet/issues/1809)) ([a39fb1b](https://github.com/ooples/AiDotNet/commit/a39fb1bae59b737904696e94b03ff2452c06c4e6))
* cache Adam per-param backing arrays across steps (+ NBEATS bias-col reuse) ([#1816](https://github.com/ooples/AiDotNet/issues/1816)) ([29d2ad4](https://github.com/ooples/AiDotNet/commit/29d2ad468a6c30bb49764a48f767f00b6d4148b7))
* **optimizer:** parallelize eager Adam Step across parameters ([#1806](https://github.com/ooples/AiDotNet/issues/1806)) ([0b3adac](https://github.com/ooples/AiDotNet/commit/0b3adac800f0804534519eec16be7f08d0deb213))
* **optimizer:** wire eager fp32 Adam step onto the shared SIMD kernel ([#1815](https://github.com/ooples/AiDotNet/issues/1815)) ([1a5125c](https://github.com/ooples/AiDotNet/commit/1a5125ca9e73f9a57e96ac9777efa5e51ff67235))

## [v0.207.0] - 2026-05-21

_Release v0.207.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.207.0

## [v0.206.0] - 2026-05-19

_Release v0.206.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.206.0

## [v0.205.0] - 2026-05-19

_Release v0.205.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.205.0

## [v0.204.0] - 2026-05-18

_Release v0.204.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.204.0

## [v0.203.0] - 2026-05-17

_Release v0.203.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.203.0

## [v0.202.0] - 2026-05-17

_Release v0.202.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.202.0

## [v0.201.0] - 2026-05-17

_Release v0.201.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.201.0

## [v0.200.0] - 2026-05-17

_Release v0.200.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.200.0

## [v0.199.0] - 2026-05-17

_Release v0.199.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.199.0

## [v0.198.0] - 2026-05-16

_Release v0.198.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.198.0

## [v0.197.0] - 2026-05-16

_Release v0.197.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.197.0

## [v0.196.0] - 2026-05-16

_Release v0.196.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.196.0

## [v0.195.0] - 2026-05-14

_Release v0.195.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.195.0

## [v0.194.0] - 2026-05-14

_Release v0.194.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.194.0

## [v0.193.0] - 2026-05-14

_Release v0.193.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.193.0

## [v0.192.0] - 2026-05-13

_Release v0.192.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.192.0

## [v0.191.0] - 2026-05-12

_Release v0.191.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.191.0

## [v0.190.0] - 2026-05-12

_Release v0.190.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.190.0

## [v0.189.0] - 2026-05-11

_Release v0.189.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.189.0

## [v0.188.0] - 2026-05-10

_Release v0.188.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.188.0

## [v0.187.0] - 2026-05-10

_Release v0.187.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.187.0

## [v0.186.0] - 2026-05-10

_Release v0.186.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.186.0

## [v0.185.0] - 2026-05-06

_Release v0.185.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.185.0

## [v0.184.0] - 2026-05-06

_Release v0.184.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.184.0

## [v0.183.0] - 2026-05-05

_Release v0.183.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.183.0

## [v0.182.0] - 2026-05-05

_Release v0.182.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.182.0

## [v0.181.0] - 2026-05-05

_Release v0.181.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.181.0

## [v0.180.0] - 2026-05-05

_Release v0.180.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.180.0

## [v0.179.0] - 2026-05-04

_Release v0.179.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.179.0

## [v0.178.0] - 2026-05-04

_Release v0.178.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.178.0

## [v0.177.0] - 2026-05-04

_Release v0.177.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.177.0

## [v0.176.0] - 2026-05-03

_Release v0.176.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.176.0

## [v0.175.0] - 2026-05-03

_Release v0.175.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.175.0

## [v0.174.0] - 2026-05-03

_Release v0.174.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.174.0

## [v0.173.0] - 2026-05-03

_Release v0.173.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.173.0

## [v0.172.0] - 2026-05-02

_Release v0.172.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.172.0

## [v0.171.0] - 2026-04-30

_Release v0.171.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.171.0

## [v0.170.0] - 2026-04-29

_Release v0.170.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.170.0

## [v0.169.0] - 2026-04-29

_Release v0.169.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.169.0

## [v0.168.0] - 2026-04-28

_Release v0.168.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.168.0

## [v0.167.0] - 2026-04-28

_Release v0.167.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.167.0

## [v0.166.0] - 2026-04-28

_Release v0.166.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.166.0

## [v0.165.0] - 2026-04-27

_Release v0.165.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.165.0

## [v0.164.0] - 2026-04-27

_Release v0.164.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.164.0

## [v0.163.0] - 2026-04-26

_Release v0.163.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.163.0

## [v0.162.0] - 2026-04-23

_Release v0.162.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.162.0

## [v0.161.0] - 2026-04-21

_Release v0.161.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.161.0

## [v0.160.0] - 2026-04-21

_Release v0.160.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.160.0

## [v0.159.0] - 2026-04-20

_Release v0.159.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.159.0

## [v0.158.0] - 2026-04-19

_Release v0.158.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.158.0

## [v0.157.0] - 2026-04-17

_Release v0.157.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.157.0

## [v0.156.0] - 2026-04-17

_Release v0.156.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.156.0

## [v0.155.0] - 2026-04-17

_Release v0.155.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.155.0

## [v0.154.0] - 2026-04-17

_Release v0.154.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.154.0

## [v0.153.0] - 2026-04-16

_Release v0.153.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.153.0

## [v0.152.0] - 2026-04-16

_Release v0.152.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.152.0

## [v0.151.0] - 2026-04-16

_Release v0.151.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.151.0

## [v0.150.0] - 2026-04-15

_Release v0.150.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.150.0

## [v0.149.0] - 2026-04-14

_Release v0.149.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.149.0

## [v0.148.0] - 2026-04-13

_Release v0.148.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.148.0

## [v0.147.0] - 2026-04-13

_Release v0.147.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.147.0

## [v0.146.0] - 2026-04-12

_Release v0.146.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.146.0

## [v0.145.0] - 2026-04-12

_Release v0.145.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.145.0

## [v0.144.0] - 2026-04-12

_Release v0.144.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.144.0

## [v0.143.0] - 2026-04-08

_Release v0.143.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.143.0

## [v0.142.0] - 2026-04-07

_Release v0.142.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.142.0

## [v0.141.0] - 2026-04-07

_Release v0.141.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.141.0

## [v0.140.0] - 2026-04-06

_Release v0.140.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.140.0

## [v0.139.0] - 2026-04-06

_Release v0.139.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.139.0

## [v0.138.0] - 2026-04-06

_Release v0.138.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.138.0

## [v0.137.0] - 2026-04-06

_Release v0.137.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.137.0

## [v0.136.0] - 2026-04-05

_Release v0.136.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.136.0

## [v0.135.0] - 2026-04-04

_Release v0.135.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.135.0

## [v0.134.0] - 2026-04-03

_Release v0.134.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.134.0

## [v0.133.0] - 2026-03-30

_Release v0.133.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.133.0

## [v0.132.0] - 2026-03-30

_Release v0.132.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.132.0

## [v0.131.0] - 2026-03-29

_Release v0.131.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.131.0

## [v0.130.0] - 2026-03-29

_Release v0.130.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.130.0

## [v0.129.0] - 2026-03-29

_Release v0.129.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.129.0

## [v0.128.0] - 2026-03-29

_Release v0.128.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.128.0

## [v0.127.0] - 2026-03-29

_Release v0.127.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.127.0

## [v0.126.0] - 2026-03-28

_Release v0.126.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.126.0

## [v0.125.0] - 2026-03-28

_Release v0.125.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.125.0

## [v0.124.0] - 2026-03-28

_Release v0.124.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.124.0

## [v0.123.0] - 2026-03-27

_Release v0.123.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.123.0

## [v0.122.0] - 2026-03-27

_Release v0.122.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.122.0

## [v0.121.0] - 2026-03-26

_Release v0.121.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.121.0

## [v0.120.0] - 2026-03-26

_Release v0.120.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.120.0

## [v0.119.0] - 2026-03-16

_Release v0.119.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.119.0

## [v0.118.0] - 2026-03-16

_Release v0.118.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.118.0

## [v0.117.0] - 2026-03-16

_Release v0.117.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.117.0

## [v0.116.0] - 2026-03-13

_Release v0.116.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.116.0

## [v0.115.0] - 2026-03-12

_Release v0.115.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.115.0

## [v0.114.0] - 2026-03-10

_Release v0.114.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.114.0

## [v0.113.0] - 2026-03-10

_Release v0.113.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.113.0

## [v0.112.0] - 2026-03-10

_Release v0.112.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.112.0

## [v0.111.0] - 2026-03-09

_Release v0.111.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.111.0

## [v0.110.0] - 2026-03-09

_Release v0.110.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.110.0

## [v0.109.0] - 2026-03-09

_Release v0.109.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.109.0

## [v0.108.0] - 2026-03-08

_Release v0.108.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.108.0

## [v0.107.0] - 2026-03-07

_Release v0.107.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.107.0

## [v0.106.0] - 2026-03-07

_Release v0.106.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.106.0

## [v0.105.0] - 2026-03-02

_Release v0.105.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.105.0

## [v0.104.0] - 2026-03-02

_Release v0.104.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.104.0

## [v0.103.0] - 2026-03-02

_Release v0.103.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.103.0

## [v0.102.0] - 2026-03-02

_Release v0.102.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.102.0

## [v0.101.0] - 2026-03-02

_Release v0.101.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.101.0

## [v0.100.0] - 2026-03-01

_Release v0.100.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.100.0

## [v0.99.0] - 2026-03-01

_Release v0.99.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.99.0

## [v0.98.0] - 2026-03-01

_Release v0.98.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.98.0

## [v0.97.0] - 2026-03-01

_Release v0.97.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.97.0

## [v0.96.0] - 2026-02-24

_Release v0.96.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.96.0

## [v0.95.0] - 2026-02-23

_Release v0.95.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.95.0

## [v0.94.0] - 2026-02-23

_Release v0.94.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.94.0

## [v0.93.0] - 2026-02-17

_Release v0.93.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.93.0

## [v0.92.0] - 2026-02-17

_Release v0.92.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.92.0

## [v0.91.0] - 2026-02-17

_Release v0.91.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.91.0

## [v0.90.0] - 2026-02-16

_Release v0.90.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.90.0

## [v0.89.0] - 2026-02-15

_Release v0.89.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.89.0

## [v0.88.0] - 2026-02-15

_Release v0.88.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.88.0

## [v0.87.0] - 2026-02-13

_Release v0.87.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.87.0

## [v0.86.0] - 2026-02-11

_Release v0.86.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.86.0

## [v0.85.0] - 2026-02-11

_Release v0.85.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.85.0

## [v0.84.0] - 2026-02-10

_Release v0.84.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.84.0

## [v0.83.0] - 2026-02-10

_Release v0.83.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.83.0

## [v0.82.0] - 2026-02-09

_Release v0.82.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.82.0

## [v0.81.0] - 2026-02-09

_Release v0.81.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.81.0

## [v0.80.0] - 2026-02-07

_Release v0.80.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.80.0

## [v0.79.0] - 2026-02-06

_Release v0.79.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.79.0

## [v0.78.0] - 2026-02-04

_Release v0.78.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.78.0

## [v0.77.0] - 2026-02-03

_Release v0.77.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.77.0

## [v0.76.0] - 2026-02-02

_Release v0.76.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.76.0

## [v0.75.0] - 2026-01-31

_Release v0.75.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.75.0

## [v0.74.0] - 2026-01-29

_Release v0.74.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.74.0

## [v0.73.0] - 2026-01-27

_Release v0.73.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.73.0

## [v0.72.0] - 2026-01-27

_Release v0.72.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.72.0

## [v0.71.0] - 2026-01-27

_Release v0.71.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.71.0

## [v0.70.0] - 2026-01-26

_Release v0.70.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.70.0

## [v0.69.0] - 2026-01-25

_Release v0.69.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.69.0

## [v0.68.0] - 2026-01-24

_Release v0.68.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.68.0

## [v0.67.0] - 2026-01-23

_Release v0.67.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.67.0

## [v0.66.0] - 2026-01-23

_Release v0.66.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.66.0

## [v0.65.0] - 2026-01-22

_Release v0.65.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.65.0

## [v0.64.0] - 2026-01-22

_Release v0.64.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.64.0

## [v0.63.0] - 2026-01-21

_Release v0.63.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.63.0

## [v0.62.0] - 2026-01-21

_Release v0.62.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.62.0

## [v0.61.0] - 2026-01-21

_Release v0.61.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.61.0

## [v0.60.0] - 2026-01-20

_Release v0.60.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.60.0

## [v0.59.0] - 2026-01-20

_Release v0.59.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.59.0

## [v0.58.0] - 2026-01-20

_Release v0.58.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.58.0

## [v0.57.0] - 2026-01-19

_Release v0.57.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.57.0

## [v0.56.0] - 2026-01-19

_Release v0.56.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.56.0

## [v0.55.0] - 2026-01-19

_Release v0.55.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.55.0

## [v0.54.0] - 2026-01-19

_Release v0.54.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.54.0

## [v0.53.0] - 2026-01-17

_Release v0.53.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.53.0

## [v0.52.0] - 2026-01-17

_Release v0.52.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.52.0

## [v0.51.0] - 2026-01-14

_Release v0.51.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.51.0

## [v0.50.0] - 2026-01-14

_Release v0.50.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.50.0

## [v0.49.0] - 2026-01-13

_Release v0.49.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.49.0

## [v0.48.0] - 2026-01-11

_Release v0.48.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.48.0

## [v0.47.0] - 2026-01-11

_Release v0.47.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.47.0

## [v0.46.0] - 2026-01-11

_Release v0.46.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.46.0

## [v0.45.0] - 2025-12-31

_Release v0.45.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.45.0

## [v0.44.0] - 2025-12-30

_Release v0.44.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.44.0

## [v0.43.0] - 2025-12-29

_Release v0.43.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.43.0

## [v0.42.0] - 2025-12-28

_Release v0.42.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.42.0

## [v0.41.0] - 2025-12-28

_Release v0.41.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.41.0

## [v0.40.0] - 2025-12-28

_Release v0.40.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.40.0

## [v0.39.0] - 2025-12-28

_Release v0.39.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.39.0

## [v0.38.0] - 2025-12-28

_Release v0.38.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.38.0

## [v0.37.0] - 2025-12-28

_Release v0.37.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.37.0

## [v0.36.0] - 2025-12-28

_Release v0.36.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.36.0

## [v0.35.0] - 2025-12-27

_Release v0.35.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.35.0

## [v0.34.0] - 2025-12-27

_Release v0.34.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.34.0

## [v0.33.0] - 2025-12-27

_Release v0.33.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.33.0

## [v0.32.0] - 2025-12-27

_Release v0.32.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.32.0

## [v0.31.0] - 2025-12-27

_Release v0.31.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.31.0

## [v0.30.0] - 2025-12-27

_Release v0.30.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.30.0

## [v0.29.0] - 2025-12-27

_Release v0.29.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.29.0

## [v0.28.0] - 2025-12-26

_Release v0.28.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.28.0

## [v0.27.0] - 2025-12-26

_Release v0.27.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.27.0

## [v0.26.0] - 2025-12-26

_Release v0.26.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.26.0

## [v0.25.0] - 2025-12-26

_Release v0.25.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.25.0

## [v0.24.0] - 2025-12-25

_Release v0.24.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.24.0

## [v0.23.0] - 2025-12-25

_Release v0.23.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.23.0

## [v0.22.0] - 2025-12-25

_Release v0.22.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.22.0

## [v0.21.0] - 2025-12-24

_Release v0.21.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.21.0

## [v0.20.0] - 2025-12-22

_Release v0.20.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.20.0

## [v0.19.0] - 2025-12-22

_Release v0.19.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.19.0

## [v0.18.0] - 2025-12-22

_Release v0.18.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.18.0

## [v0.17.0] - 2025-12-22

_Release v0.17.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.17.0

## [v0.16.0] - 2025-12-22

_Release v0.16.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.16.0

## [v0.15.0] - 2025-12-21

_Release v0.15.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.15.0

## [v0.14.0] - 2025-12-21

_Release v0.14.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.14.0

## [v0.13.0] - 2025-12-21

_Release v0.13.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.13.0

## [v0.12.0] - 2025-12-20

_Release v0.12.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.12.0

## [v0.11.0] - 2025-12-19

_Release v0.11.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.11.0

## [v0.10.0] - 2025-12-17

_Release v0.10.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.10.0

## [v0.9.0] - 2025-12-17

_Release v0.9.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.9.0

## [v0.8.0] - 2025-12-15

_Release v0.8.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.8.0

## [v0.7.0] - 2025-12-14

_Release v0.7.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.7.0

## [v0.6.0] - 2025-12-14

_Release v0.6.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.6.0

## [v0.5.0] - 2025-12-14

_Release v0.5.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.5.0

## [v0.4.0] - 2025-12-14

_Release v0.4.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.4.0

## [v0.3.0] - 2025-12-11

_Release v0.3.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.3.0

## [v0.2.0] - 2025-11-15

_Release v0.2.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.2.0

## [v0.1.0] - 2025-11-12

_Release v0.1.0_

See https://github.com/ooples/AiDotNet/releases/tag/v0.1.0

## [v0.0.5-preview] - 2023-10-16 (pre-release)

See https://github.com/ooples/AiDotNet/releases/tag/v0.0.5-preview

## [v0.0.3-preview] - 2023-09-25 (pre-release)

See https://github.com/ooples/AiDotNet/releases/tag/v0.0.3-preview

## [v0.0.1-preview] - 2023-09-23 (pre-release)

See https://github.com/ooples/AiDotNet/releases/tag/v0.0.1-preview
