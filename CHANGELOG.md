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

## [0.233.1](https://github.com/ooples/AiDotNet/compare/v0.233.0...v0.233.1) (2026-09-26)


### Bug Fixes

* **ci:** stop the heavy-timeout lane OOM-killing its own runner ([3f9bbde](https://github.com/ooples/AiDotNet/commit/3f9bbde222daa6d5f9e4dae9ceba47080b35ba51))
* persist deep gaussian process layer state and drop the shallow gp clone ([4d27188](https://github.com/ooples/AiDotNet/commit/4d2718825f06b5429163fec124197853d06222ae))
* **release:** cap build parallelism so the release runner survives the compile ([3db9675](https://github.com/ooples/AiDotNet/commit/3db9675b0c127dcae17b87dcf73df41716c77d7d))
* **release:** cap build parallelism so the release runner survives the compile ([7b2f204](https://github.com/ooples/AiDotNet/commit/7b2f204c100722a0f3043623e07bd3b3fe82a7ff))
* **video:** reject raft frame pairs with the wrong stacked channel count ([ba798ad](https://github.com/ooples/AiDotNet/commit/ba798ad1e5a325941086e375fceb18ecde10dfa0))
* **video:** route VideoMAE pretraining through its decoder and declare TimeSformer's clip input ([a9d755e](https://github.com/ooples/AiDotNet/commit/a9d755ee24302e2de82606bf50a26d9108984f59))


### Refactoring

* **video:** name videomae's target-normalization offset; assert videoclip shape ([028e606](https://github.com/ooples/AiDotNet/commit/028e6061474930c711a6fc2ce8b82bffe5e7e703))


### Documentation

* **activation:** clarify temperature conversion ([3eb874f](https://github.com/ooples/AiDotNet/commit/3eb874f666587669f6c5e5712fd8c15171e7ce40))

## [0.233.0](https://github.com/ooples/AiDotNet/compare/v0.232.0...v0.233.0) (2026-09-25)


### Features

* **cv:** detection/OCR test families, evaluation metrics, and train/clone/predict fixes ([7c9cf1c](https://github.com/ooples/AiDotNet/commit/7c9cf1c97e34c87f61506652beb6fcdf1e8beea6))


### Bug Fixes

* **build:** drop the missing win-arm asset SQLitePCLRaw 2.1.12 still declares ([bdf9b8d](https://github.com/ooples/AiDotNet/commit/bdf9b8d55026f9bc2f2eb65b16e44df0377fd6d5))
* **cv:** register swin stage tensors without claiming the stage list as state ([9daac21](https://github.com/ooples/AiDotNet/commit/9daac212bf8fb482e21ff2836764d658455dc363))
* **cv:** train yolov10's one-to-many head through the generic train path ([9831688](https://github.com/ooples/AiDotNet/commit/9831688077ee2b25a2f08bc66cdf9da10627ed63))
* **generators:** copy param model before setting writer-only flag ([98b964e](https://github.com/ooples/AiDotNet/commit/98b964e697af85fe6d090606c48faec345cbf61e))
* **generators:** omit null json and expression state from later constructors ([d5eab65](https://github.com/ooples/AiDotNet/commit/d5eab656184f5b38238f28d2b2840783b5239589))
* **metadata:** defer blip and videoclip onnx model data to a provider ([2a747cb](https://github.com/ooples/AiDotNet/commit/2a747cbb0a27991a0221407d03bced7a3f374db4))
* **metadata:** stop 290 models serializing their weights on every build ([7dfc96c](https://github.com/ooples/AiDotNet/commit/7dfc96ced43e15012b73524bca7f4f38c6aa88cf))
* **release:** add a dispatch path that publishes an existing release tag ([72c262a](https://github.com/ooples/AiDotNet/commit/72c262a413c6e0d243250cfc36ac9d32be5d9a85))
* **release:** build only the packed projects, and size the job timeout from measurement ([7ab4a80](https://github.com/ooples/AiDotNet/commit/7ab4a80287ec3159cbe5319bac9968a3c6865a34))
* **release:** build only the packed projects, and size the job timeout from measurement ([86053a3](https://github.com/ooples/AiDotNet/commit/86053a388f5fc045a1b6c8201b6423ab83ba0ded))
* **release:** let a dispatched recovery run reach build and publish ([55246d8](https://github.com/ooples/AiDotNet/commit/55246d88d4e644ed8b060ac12ec30cc9215549d1))
* **release:** let a dispatched recovery run reach build and publish ([829ebc0](https://github.com/ooples/AiDotNet/commit/829ebc02047853b8041d2b98142c69430be1b259))
* **release:** never republish or replace a published package ([ae3a57a](https://github.com/ooples/AiDotNet/commit/ae3a57a1eb78516ea332c4203f5c069a06541c4c))
* **release:** respect cancellation in the dispatched build and publish jobs ([d550de6](https://github.com/ooples/AiDotNet/commit/d550de67ab6e88d516103ec163e9fa36fa776959))
* **tests:** fix 3 flaky tests and 2 never-run sweeps (DBN, CompiledMlp, Integration D OOM) ([2b5c60f](https://github.com/ooples/AiDotNet/commit/2b5c60ffe6041561196daad212fed35c9eaf525d))


### Performance

* **cv:** pool roialign bins with one batched product instead of a broadcast mask ([c5ee447](https://github.com/ooples/AiDotNet/commit/c5ee44703cecac6f041b1450ad3f4af7fcfd55b3))
* **release:** check NuGet with HEAD instead of downloading each package ([25ca56b](https://github.com/ooples/AiDotNet/commit/25ca56be5d8909d526a71d48cb51aef5acfff3d1))

## [0.232.0](https://github.com/ooples/AiDotNet/compare/v0.231.0...v0.232.0) (2026-09-24)


### Features

* **#2090:** phase 2 — sequence models read their Options ([0ca40af](https://github.com/ooples/AiDotNet/commit/0ca40af6e2c73776b96b07e233b9526ffd3f8812))
* **#2090:** phase 2 complete — all 17 sequence models, ratchet 1004 -&gt; 977 ([671836a](https://github.com/ooples/AiDotNet/commit/671836a347436f4b023ebe3f02a86ea333ad0686)), closes [#2090](https://github.com/ooples/AiDotNet/issues/2090)
* **#2090:** phase 3 — vision-language models read their Options ([085db69](https://github.com/ooples/AiDotNet/commit/085db6915d7b21c214ba5e24a016b39d7b56360e))
* allow specialized trading environments to override transition execution ([1a3e02f](https://github.com/ooples/AiDotNet/commit/1a3e02fdea4f76e62c2110b9d5684281e204ca0e))
* allow specialized trading transition execution ([5c281c3](https://github.com/ooples/AiDotNet/commit/5c281c3aa7f0c9821376d67794dc1c6266784537))
* **ci:** add conservative old-new dependency selection with negative controls ([52876fe](https://github.com/ooples/AiDotNet/commit/52876fe686b674f0eabf41149194223763fd2ee9))
* **ci:** bind complete configuration bodies and private initializers ([9da2fea](https://github.com/ooples/AiDotNet/commit/9da2fea6489876daf3c463c2671b7b4d8acac896))
* **ci:** bind concrete numeric and scoped runtime effects ([a7524d5](https://github.com/ooples/AiDotNet/commit/a7524d533df79feae9d0c5dd8b193ecabca69acb))
* **ci:** bind conditional owner effects and startup inputs ([ad02585](https://github.com/ooples/AiDotNet/commit/ad025859d0329cb1a5f30ba4cafa93b5045791ad))
* **ci:** bind default startup flow and successful CPU reset ([8e0093b](https://github.com/ooples/AiDotNet/commit/8e0093b164678fb026e1300a665bbd29d7762066))
* **ci:** bind method execution plans to discovered tests and binaries ([200e4dd](https://github.com/ooples/AiDotNet/commit/200e4dd0eace1eeb1de0d2eae2972e9bd1ca4fda))
* **ci:** bind reviewed assertion contracts to runtime evidence ([86f6e49](https://github.com/ooples/AiDotNet/commit/86f6e49c28505682f33c933eb499c2b9a3995727))
* **ci:** bind reviewed crypto startup and reject shadow runtimes ([4e382f9](https://github.com/ooples/AiDotNet/commit/4e382f9edba31e3129aafa68b32dfc7bbf715d12))
* **ci:** bind runtime effects and constructor input contracts ([4dd21eb](https://github.com/ooples/AiDotNet/commit/4dd21eb3bcf18e00203394049cbefc14ff4a36c4))
* **ci:** bind source changes to verified local execution reuse ([4a4f950](https://github.com/ooples/AiDotNet/commit/4a4f950b63b09127ad5bdbe97a47452c5dd24a73))
* **ci:** bind startup branch inputs and review GPU opt-out effects ([0635476](https://github.com/ooples/AiDotNet/commit/0635476bb7b533688162352fd0642713e1d1564f))
* **ci:** join body isolation across the exact workload inventory ([9d5a457](https://github.com/ooples/AiDotNet/commit/9d5a4574c3d958ab0524a8989d73aeb030bc39ca))
* **ci:** measure what the shard map can actually select ([6c66601](https://github.com/ooples/AiDotNet/commit/6c666015075754cfac51e355b096dbd4e8b1362c))
* **ci:** preserve concrete base and shared-field bindings ([ee43fae](https://github.com/ooples/AiDotNet/commit/ee43fae97ca6fc120f7e648446c2d0576922f208))
* **ci:** prototype owned runtime effects with full-run controls ([d545546](https://github.com/ooples/AiDotNet/commit/d545546fe87781a26c3d2837a0ceba18edbf95cf))
* **ci:** validate changed-base execution reuse partitions ([4b0534d](https://github.com/ooples/AiDotNet/commit/4b0534dd01a4e097ad8ed8542f067a2c3e48a3a5))
* **ci:** verify cross-assembly impact and authenticated workflow imports ([9b92c9c](https://github.com/ooples/AiDotNet/commit/9b92c9c83ea2ba03a284bd93728c64ac780aa8de))
* **ci:** verify observed owners and bind lifecycle profiles ([978eeab](https://github.com/ooples/AiDotNet/commit/978eeab040e12ca720d99a63a0002a0710a90e45))
* **ci:** verify observed trial scopes and slot initialization ([1f42fab](https://github.com/ooples/AiDotNet/commit/1f42fab10371331578259c41310ecb49648abc4d))
* **ci:** verify trial hooks and exact CPU reset observations ([2f993c2](https://github.com/ooples/AiDotNet/commit/2f993c27355aea0b93bc7a95e06da4dc5bd13513))
* declare paper optimizers for diffusion-ts, ppo, bytetrack and a3c ([85b2d18](https://github.com/ooples/AiDotNet/commit/85b2d1837899461f3f89af1fbc509f3ceb7c7a7f)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for four physics-informed models ([a2f0894](https://github.com/ooples/AiDotNet/commit/a2f0894b563b12683dcf1780d93ec13b732ef182)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for sam 2.1, efficienttam, pixellm and deva ([e6b3c10](https://github.com/ooples/AiDotNet/commit/e6b3c10a9082101e29ad85c06e4695e86d9165d1)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for seven papers recovered from doi-form urls ([cdcf9dc](https://github.com/ooples/AiDotNet/commit/cdcf9dceb172cea26be01e4ce41792bcaa005b6a)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for show-o, video-llava and three more ([660ea6f](https://github.com/ooples/AiDotNet/commit/660ea6fcdb5300fc8ce8140422a30adfe87e0086)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for six unified and 3d vision-language models ([f9dae37](https://github.com/ooples/AiDotNet/commit/f9dae37fdfba6ecb5ab10a02fa61896dd5c26051)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for the last seven papers with stated recipes ([d67b8a6](https://github.com/ooples/AiDotNet/commit/d67b8a6b8522d7c6dde16ac57e2ad330feaa9366)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for two code models and four audio models ([39211c7](https://github.com/ooples/AiDotNet/commit/39211c7f3d384585c1fbb191d97ae18e374a7a9a)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* declare paper optimizers for univnet, informer, n-beats and two ner models ([f3f5ecf](https://github.com/ooples/AiDotNet/commit/f3f5ecf1132f27a0c2e42da83549583ee8143fea)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* **finance:** add a weight-resolution seam to PortfolioManagerEnvironment ([3d60407](https://github.com/ooples/AiDotNet/commit/3d604071463a310115dd805f31d79c0bea93dd8c))
* **finance:** add a weight-resolution seam to PortfolioManagerEnvironment ([83807f1](https://github.com/ooples/AiDotNet/commit/83807f1397a16ddf71546ea8aedaac8734a90391))
* **finance:** configurable SAC head, friction overrides and loud checkpoint failure ([48333d3](https://github.com/ooples/AiDotNet/commit/48333d3a034fbb45141d415ad2972b2b20750977))
* **models:** declare nine video super-resolution recipes ([#1928](https://github.com/ooples/AiDotNet/issues/1928) batch 4) ([611bc35](https://github.com/ooples/AiDotNet/commit/611bc358c022f9ad687a5141f74e3419a45e4d66))
* **models:** declare paper training recipes, batch 2 ([#1928](https://github.com/ooples/AiDotNet/issues/1928)) ([e98cf42](https://github.com/ooples/AiDotNet/commit/e98cf42bfbaebbbac553b28cfae987cebd592000))
* **models:** declare paper training recipes, batch 3 ([#1928](https://github.com/ooples/AiDotNet/issues/1928)) ([cee9e4d](https://github.com/ooples/AiDotNet/commit/cee9e4da8769dcf3acadf44885ef977e4e5e5583))
* **models:** declare paper-faithful training recipes ([#1928](https://github.com/ooples/AiDotNet/issues/1928) batch 5) ([b8af12a](https://github.com/ooples/AiDotNet/commit/b8af12a706044765f59d5e6e663ff77118fd340e))
* **models:** declare paper-faithful training recipes ([#1928](https://github.com/ooples/AiDotNet/issues/1928) batch 6) ([11af811](https://github.com/ooples/AiDotNet/commit/11af81134f8515c6f9f7bb5989b8925740a6f74f))
* **models:** train models the way their papers say, and report when they do not ([#1928](https://github.com/ooples/AiDotNet/issues/1928)) ([96056cb](https://github.com/ooples/AiDotNet/commit/96056cb000eacdcb56327911613393d1e49aab48))
* **optimizers:** let models declare the optimizer settings their paper specifies ([#1928](https://github.com/ooples/AiDotNet/issues/1928) mechanism) ([bd8c032](https://github.com/ooples/AiDotNet/commit/bd8c03271c2c6d350d9abd8742fe4b58e823a59f))
* **options:** migrate enum and string parameters, ratchet 875 -&gt; 861 ([243f6f6](https://github.com/ooples/AiDotNet/commit/243f6f6e135b139fc9edef9e097c0279c2e9d2fe)), closes [#2090](https://github.com/ooples/AiDotNet/issues/2090)
* **options:** phase 2 complete — all 17 sequence models, ratchet 1004 -&gt; 977 ([94abf52](https://github.com/ooples/AiDotNet/commit/94abf52313b9868c5cfd7390e9bf45e135fb188c)), closes [#2090](https://github.com/ooples/AiDotNet/issues/2090)
* **options:** phase 3 — vision-language models, ratchet 977 -&gt; 875 ([e41b11e](https://github.com/ooples/AiDotNet/commit/e41b11e32b315dba2e59b52f1d3dee04dca2ac79)), closes [#2090](https://github.com/ooples/AiDotNet/issues/2090)
* **rl:** legal-action masking, carried by the environment ([8b83bc4](https://github.com/ooples/AiDotNet/commit/8b83bc4537e014cac0b7faab5797569e5d674310))
* **rl:** legal-action masking, enforced at every agent selection site ([02e9f45](https://github.com/ooples/AiDotNet/commit/02e9f45045384c7e34dc428077b6ea31cbb78410))
* **rl:** mask every agent selection site, incl. the FinRL wrapper ([8d9c6b3](https://github.com/ooples/AiDotNet/commit/8d9c6b33c61b627d15e8b9e274555c32f4d21ab4))
* **testing:** add tape reachability probe that catches detached training terms ([f0b9026](https://github.com/ooples/AiDotNet/commit/f0b902655bba901915636c6caedfd5309175df86))
* **website:** licence Token Optimizer as a product, with savings receipts ([15ebf3c](https://github.com/ooples/AiDotNet/commit/15ebf3c883782b6eb7de4c5405c5b38cbeed6ac9))
* **website:** licence Token Optimizer as a product, with savings receipts ([414a1a8](https://github.com/ooples/AiDotNet/commit/414a1a876fbb384320dd1b40f319b2fbbf51b895))


### Bug Fixes

* **active-learning:** make RandomSampling's seed change which samples it picks ([e9d174a](https://github.com/ooples/AiDotNet/commit/e9d174ab5bf16df3c09e9b0b4981480ace1cb68d))
* add the missing declarations for octo, tabdpt and abinet ([98d9dfd](https://github.com/ooples/AiDotNet/commit/98d9dfdf5166fb258db8717ca6d684dd57a673ac)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* address masked policy numeric and lifetime review ([0e3fdd4](https://github.com/ooples/AiDotNet/commit/0e3fdd416e1a1cb0281a886632c54e5fa8b9c833))
* address review feedback on vision-language options surface ([0d496e5](https://github.com/ooples/AiDotNet/commit/0d496e52bb7e80ad8ad1f8999a0a23ea222d5121))
* **audio:** avoid virtual layer dispatch during batch-five construction ([130d9e6](https://github.com/ooples/AiDotNet/commit/130d9e63902b80b680b5434d278b2152104e074f))
* **audio:** avoid virtual layer dispatch during batch-five construction ([59e7dd9](https://github.com/ooples/AiDotNet/commit/59e7dd9d922ad62ef40c6390d7dcd89e28ab649b))
* **audio:** avoid virtual layer dispatch during batch-four construction ([4464193](https://github.com/ooples/AiDotNet/commit/4464193cf16b04e580d878689b851ac2707195ab))
* **audio:** avoid virtual layer dispatch during batch-four construction ([4c02b02](https://github.com/ooples/AiDotNet/commit/4c02b02d79673fd6b8d60cac19ab629b9518ea1b))
* **audio:** avoid virtual layer dispatch during batch-four construction ([e90307c](https://github.com/ooples/AiDotNet/commit/e90307ca6f2a49fbfefb081c1c6ba87843579437))
* **audio:** avoid virtual layer dispatch during batch-six construction ([14386f1](https://github.com/ooples/AiDotNet/commit/14386f17b985cac46b880fab3f31e14430cb9b7e))
* **audio:** train music flamingo at its declared learning rate ([7a33449](https://github.com/ooples/AiDotNet/commit/7a33449d61ed9a550ae24cd7615bad27ffa8d2d9))
* **audio:** use a floating-point frame rate for BeatTracker tempo lag bounds ([502176b](https://github.com/ooples/AiDotNet/commit/502176b4c31164d06469a3613b09c853e965e9be))
* **audio:** widen the class-weight denominator in AudioClassifierBase ([9b174dc](https://github.com/ooples/AiDotNet/commit/9b174dc195cc3d56c31a7145b4083e8017086e46))
* backpropagate the rainbow dqn update instead of index-modulo drift ([9c5ae50](https://github.com/ooples/AiDotNet/commit/9c5ae50ba44aa64f028b3c53fbcc9a11175938ec))
* **blip:** validate ONNX graph options and embedding contracts ([c74ecd7](https://github.com/ooples/AiDotNet/commit/c74ecd7fdd579ca40cd335ce42017a68d3302aee))
* build each model's options once, not twice ([#2228](https://github.com/ooples/AiDotNet/issues/2228)) ([b3c786f](https://github.com/ooples/AiDotNet/commit/b3c786ffc975f6cec0502154e23b18e9936a8559))
* build each model's options once, not twice ([#2228](https://github.com/ooples/AiDotNet/issues/2228)) ([cfef411](https://github.com/ooples/AiDotNet/commit/cfef411c0b3a986bdb88fade33c352c29d1dced5))
* **build:** align EF Core packages with SQLite 10.0.12 ([0fd4ada](https://github.com/ooples/AiDotNet/commit/0fd4ada0c45b1e51913887aed91893c25fd0a107))
* cap the target-dependence repeats for bounded generated training fixtures ([8409679](https://github.com/ooples/AiDotNet/commit/84096799bc6ca0ed09be8245ea1a3ac25193dc33))
* **ci:** a missing retry baseline must not fail the shard ([66e4a8e](https://github.com/ooples/AiDotNet/commit/66e4a8e76fdc61cf72d56af385b2bb03bac3004f))
* **ci:** accept absent optional delta map inputs ([2a878c5](https://github.com/ooples/AiDotNet/commit/2a878c52d9dcbfcc5e2376cd3556b8575f988bb9))
* **ci:** accept absent optional delta map inputs ([921c968](https://github.com/ooples/AiDotNet/commit/921c968b1fd9f40e94b72131d75790b0ba8d24c9))
* **ci:** add the two repaired model-shape sweeps as selectable shards ([d1e7fc1](https://github.com/ooples/AiDotNet/commit/d1e7fc15dd03f0f6d1aa38b5a09799cb8ce479ae))
* **ci:** address PR scoping and directory owner review gaps ([2905be0](https://github.com/ooples/AiDotNet/commit/2905be09afa8efe50cd67ff95f8705ae7a7c0e92))
* **ci:** bind CPU startup logging environment ([8c0b94f](https://github.com/ooples/AiDotNet/commit/8c0b94ff991eabfa86892c503e403d7f0cb3c90b))
* **ci:** bind delta reuse to consistent certificates and safe imports ([b5e7cc1](https://github.com/ooples/AiDotNet/commit/b5e7cc18afd3de970a3a7c226bf1af4c5188073d))
* **ci:** bind execution evidence and preserve generated source maps ([6e302c5](https://github.com/ooples/AiDotNet/commit/6e302c542fa1e97412add5573a15372c125e6636))
* **ci:** bind reuse evidence and harden workflow regression controls ([d550758](https://github.com/ooples/AiDotNet/commit/d550758a6682eaafb4df2c534f6adc16e49cc9f1))
* **ci:** bind source maps to embedded compiler source bytes ([c59bc2a](https://github.com/ooples/AiDotNet/commit/c59bc2a566bdb64c4fc7ea354672d9862f0edadc))
* **ci:** block superseded push retries before dependent validation ([d2fa48d](https://github.com/ooples/AiDotNet/commit/d2fa48d4a3ec8f5d9c8f757a3e7e04d68f93da2c))
* **ci:** bound collection and abstract fixture dependency routing ([8420e0f](https://github.com/ooples/AiDotNet/commit/8420e0ff2955e6f2bc34917defeedb57592917f7))
* **ci:** bound the delta-map step, decline partial reuse missing an artifact, harden reads ([7e4c25f](https://github.com/ooples/AiDotNet/commit/7e4c25ff41fbd90b632556af384a148209f3304d))
* **ci:** bound workflow import and reject malformed evidence ([ca0c86b](https://github.com/ooples/AiDotNet/commit/ca0c86bd6a8e775fb632da654b03b1282b60fc39))
* **ci:** build each project once for net10.0 in SonarCloud so analysis can finish ([15f981e](https://github.com/ooples/AiDotNet/commit/15f981ebfc01c70ce04196793991107b9946f7b2))
* **ci:** build each project once for net10.0 in SonarCloud so analysis can finish ([8b1bf72](https://github.com/ooples/AiDotNet/commit/8b1bf724414a94597a7cc57eac517e53b87022a2))
* **ci:** build the two model-shape sweeps in the worker child process ([828509c](https://github.com/ooples/AiDotNet/commit/828509c788a2da0c92ca11ec27ade825cdc13d1d))
* **ci:** checkout validation reuse resolver ([52324b8](https://github.com/ooples/AiDotNet/commit/52324b846c1bcbfc75b14bb7a42a127abb1eb027))
* **ci:** checkout validation reuse resolver ([566fb7f](https://github.com/ooples/AiDotNet/commit/566fb7f1911bea84def6dd3a2ce207b27a6210b8))
* **ci:** compile a build-only change without running every test ([ef47a66](https://github.com/ooples/AiDotNet/commit/ef47a661189cfca2b8f1bd46cce9931ab5dcb60d))
* **ci:** consume worker tickets once and invalidate unsupported ValueTask evidence ([e5fb9eb](https://github.com/ooples/AiDotNet/commit/e5fb9eb180e885f4d20f9b45ff8fd99cda8d5d3b))
* **ci:** count the whole shard universe when measuring map coverage ([b17ff54](https://github.com/ooples/AiDotNet/commit/b17ff54c657fd61e4ae7b2c14dabb1d3a403a871))
* **ci:** defer pending merge validation and validate all delta effects ([9ea50f7](https://github.com/ooples/AiDotNet/commit/9ea50f7f55012dab21253c65adc10b1240457cd8))
* **ci:** defer pending merge validation and validate every delta-affected shard ([7c43269](https://github.com/ooples/AiDotNet/commit/7c4326979dc89487e90c99f9c77430c2903c777a))
* **ci:** define Sonar validation mode in its job ([c78a6f8](https://github.com/ooples/AiDotNet/commit/c78a6f8287866e431357d53a154a242a890fa02c))
* **ci:** define Sonar validation mode in its job ([8d816ca](https://github.com/ooples/AiDotNet/commit/8d816ca17da7dece7d526a543bfc0053e17ca7fd))
* **ci:** eliminate COW and Integration-D shard regressions ([4ee6a92](https://github.com/ooples/AiDotNet/commit/4ee6a92a2ba72ff461a5112b10ad5fdb45e39f4c))
* **ci:** execute resolver fixture on Linux ([58a71a6](https://github.com/ooples/AiDotNet/commit/58a71a6a4e8e205b2725ea69c26a04cad0b82e23))
* **ci:** exit 0 after a valid selection, keep result shape uniform, read multi-line modifiers ([dce65d3](https://github.com/ooples/AiDotNet/commit/dce65d3263237426ac9263221e4140654830d586))
* **ci:** expand the shard env vars that were left inside single quotes ([d04777c](https://github.com/ooples/AiDotNet/commit/d04777c99076a73e03a65e07672072078b97f637))
* **ci:** expose scheduling inputs and test strict delta arguments ([99bc8ee](https://github.com/ooples/AiDotNet/commit/99bc8ee9ec0ac50d1f8cc2bb6234d1ed80d69ba4))
* **ci:** extract the shard run block to a script, under the expression limit ([93e7ccb](https://github.com/ooples/AiDotNet/commit/93e7ccb55be587cce386075ac120b354c1b501d9))
* **ci:** extract the shard run block to a script, under the expression limit ([2774f83](https://github.com/ooples/AiDotNet/commit/2774f83048b844b1ffc1abcede8a57e77dc34c95))
* **ci:** finish retiring the sweep jobs from the gate and its contract ([7bd5ec3](https://github.com/ooples/AiDotNet/commit/7bd5ec3640b52a15ed7e6dec2151104b2dd679b6))
* **ci:** guard the reduction ratchet on a map actually being built ([adf55e7](https://github.com/ooples/AiDotNet/commit/adf55e7798566485698c7ee497e1c894e42124bf))
* **ci:** harvest coverage runs whose manifest differs from master ([02206d2](https://github.com/ooples/AiDotNet/commit/02206d2190eba14ea9160bb0f7108d305ce491ef))
* **ci:** harvest maps across manifest changes and keep selection safe under drift ([5832c1f](https://github.com/ooples/AiDotNet/commit/5832c1f15c395b6f93b30d0f1c69ce718a42cc7a))
* **ci:** honor timing noise at the history envelope boundary ([8decd96](https://github.com/ooples/AiDotNet/commit/8decd96cd2fcdaf18242778748408f696de3fd8b))
* **ci:** keep build-only changes out of the full matrix, and two ratchet review fixes ([9da2ebd](https://github.com/ooples/AiDotNet/commit/9da2ebde0812c24da80a516532524a318561c9c0))
* **ci:** keep caller escape and assertion effects unresolved ([244e459](https://github.com/ooples/AiDotNet/commit/244e459464536bafd752d75603fad7ce91d6d89f))
* **ci:** keep independently tested transport changes selective ([04fe1c2](https://github.com/ooples/AiDotNet/commit/04fe1c29982f31c3fae48e09c418b8a8dfe5240e))
* **ci:** keep native declaration identity stable across body edits ([8a1b0c9](https://github.com/ooples/AiDotNet/commit/8a1b0c9d7fc0bf4dc79cc25eb6a52fed935d1f36))
* **ci:** keep selection reduced and safe when the shard manifest drifts ([e04dfcc](https://github.com/ooples/AiDotNet/commit/e04dfcca5a91a61b9b2fd185b94c1c9cbb43ede5))
* **ci:** let a retired shard stop wedging map certification ([9b2f31f](https://github.com/ooples/AiDotNet/commit/9b2f31f26036f2ecd6a434a6baef7231e12e3100))
* **ci:** make sonarcloud advisory and run its analysis only nightly ([19ff561](https://github.com/ooples/AiDotNet/commit/19ff561332415ba30c9049138a2ce04804ca1931))
* **ci:** make sonarcloud advisory and run its analysis only nightly ([63c07b5](https://github.com/ooples/AiDotNet/commit/63c07b5203341ed13634b289afde66328f2bba35))
* **ci:** make the selection-reduction ratchet advisory so it cannot refuse a map ([17901af](https://github.com/ooples/AiDotNet/commit/17901af162cab4dce8b79efcd30de8916e47d7e4))
* **ci:** model xunit lifecycle and bound managed dependencies ([4585b22](https://github.com/ooples/AiDotNet/commit/4585b22c53af2bf272df32dd226a7d07ca05e724))
* **ci:** move the nightly-only deferral into a tested function so the workflow loads ([6d03b3b](https://github.com/ooples/AiDotNet/commit/6d03b3b4f101afbe00eaddef982d9f29d2679acc))
* **ci:** never defer a sweep shard that the change itself requires ([5e01b42](https://github.com/ooples/AiDotNet/commit/5e01b425dcad8fcd8447321cfa27da76c2b3e019))
* **ci:** omit artifact import metadata for empty import sets ([42cec36](https://github.com/ooples/AiDotNet/commit/42cec362ed00fd136cf780ad476742ef79a0355e))
* **ci:** parse comments and raw fences inside interpolation holes ([04e202c](https://github.com/ooples/AiDotNet/commit/04e202c5e58f8988d1b65218aa8ba3af03563546))
* **ci:** preserve always-run shards without instrumentation ([9994c3a](https://github.com/ooples/AiDotNet/commit/9994c3a6b3f55daf9e3062569c8d64d15d15da88))
* **ci:** preserve required CodeQL status ([9de04b8](https://github.com/ooples/AiDotNet/commit/9de04b86edd911b7983df54edcac7e6f60adf572))
* **ci:** preserve source paths across renames ([5bb35f3](https://github.com/ooples/AiDotNet/commit/5bb35f3a3a32e05c270f06c91eb4a3fa0596bb3a))
* **ci:** preserve unchanged map certification ([c46adac](https://github.com/ooples/AiDotNet/commit/c46adac43d26a7ca483cb1995139bf9ac2a53385))
* **ci:** preserve xUnit lifecycle and revoke stale attribution reports ([1172f87](https://github.com/ooples/AiDotNet/commit/1172f87fcfbb42edb14fb6f89ec57ffd0369d7ba))
* **ci:** prove shard selection and merge reuse ([8db211e](https://github.com/ooples/AiDotNet/commit/8db211e21a56e4b69d4f21f470e4a240cc11d591))
* **ci:** re-certify the map on merge, tier build-only changes, and let the nightly carry ([3f9d6f0](https://github.com/ooples/AiDotNet/commit/3f9d6f075171c7535c55879d4fe0645bddf7d8ca))
* **ci:** record exact case outcomes and avoid large-assembly metadata overflow ([0f2a6bb](https://github.com/ooples/AiDotNet/commit/0f2a6bb9b9a96a871bba0344d866975ec78c3671))
* **ci:** reject a coverage run whose manifest is missing or not a list ([a1f51a9](https://github.com/ooples/AiDotNet/commit/a1f51a92ec90e7d6d24a1dcc9e87dc641cfc5f17))
* **ci:** reject ambiguous collection definitions ([545fba8](https://github.com/ooples/AiDotNet/commit/545fba880772f9898367a9220871875ce98f87eb))
* **ci:** release canceled workflow runners ([9aa0142](https://github.com/ooples/AiDotNet/commit/9aa0142436e1c4be81c50871f52ac76a820ed774))
* **ci:** remove the 45 retired auxiliary shard entries ([c3fdaa2](https://github.com/ooples/AiDotNet/commit/c3fdaa2466771916053da301b2670f2a83087fb9))
* **ci:** restrict attribute lifecycle roots to runner hooks ([cb946ac](https://github.com/ooples/AiDotNet/commit/cb946ac35d50bda91e35088c5a98d1a5cf05d88c))
* **ci:** resume verified artifact ranges after interrupted transfers ([527cbc2](https://github.com/ooples/AiDotNet/commit/527cbc2b9bf1b0f4edb96947c2e8db478820a13c))
* **ci:** retain imported ledger evidence in partial workload plans ([1dcfa45](https://github.com/ooples/AiDotNet/commit/1dcfa45bdc86a5bdfeb510ea0eeb203f7aff7dca))
* **ci:** retain mandatory shards in non-runtime historical audits ([d0930d2](https://github.com/ooples/AiDotNet/commit/d0930d2a4084074abc7d27335315e5953f43ae42))
* **ci:** retry transient analysis API failures ([b02e758](https://github.com/ooples/AiDotNet/commit/b02e7581ac23a24467979f1122ed37c11bcca434))
* **ci:** reuse exact successful plans across open dependencies ([89d24a9](https://github.com/ooples/AiDotNet/commit/89d24a9347199233cd0ec0bc7a8e9884a8fbd41d))
* **ci:** reuse PR validation after merging behind master; re-run only affected shards ([eee7ea6](https://github.com/ooples/AiDotNet/commit/eee7ea6000923cfae17645e5508696b26efd92e3))
* **ci:** reuse PR validation after merging behind master; re-run only affected shards ([b15f72e](https://github.com/ooples/AiDotNet/commit/b15f72ee451e356502cd78a314ecff80a38f7f28))
* **ci:** route CI policy edits by runtime execution boundaries ([74f9b79](https://github.com/ooples/AiDotNet/commit/74f9b794a3890c6539e333f45bdb636068603e64))
* **ci:** route legacy unit-test namespaces through the remaining shard ([0a19106](https://github.com/ooples/AiDotNet/commit/0a1910613c5f3ce1ce6e7c32c7b2915e5fa1ca64))
* **ci:** route non-runtime changes without full validation ([fdee6be](https://github.com/ooples/AiDotNet/commit/fdee6bea1ee66c68147ba2cf275c32f725f9fb8e))
* **ci:** run an unmapped ordinary shard instead of escalating every pr ([0c5b861](https://github.com/ooples/AiDotNet/commit/0c5b8613ae7364ebd55adf9db9191ef4bd7d47ca))
* **ci:** run model sweeps and conformance windows as selectable shards ([3774a82](https://github.com/ooples/AiDotNet/commit/3774a82066ce3b2ec1a825f6997ac9e67477a6b5))
* **ci:** run model sweeps and conformance windows as selectable shards ([130952b](https://github.com/ooples/AiDotNet/commit/130952b2618449bc0ef3ff1c2c831095f66a95d0))
* **ci:** run sweep and conformance shards nightly instead of on every pull request ([b2f5cb6](https://github.com/ooples/AiDotNet/commit/b2f5cb688e203329c62b53f0ec7ae2f84d2bd2f6))
* **ci:** run sweep and conformance shards nightly instead of on every pull request ([e7967d0](https://github.com/ooples/AiDotNet/commit/e7967d0c42c67c99b5b2a8f00eb93dbf159d2c3e))
* **ci:** safely select sweep and conformance workloads ([b19ed7a](https://github.com/ooples/AiDotNet/commit/b19ed7abc483e04a2231dadba12ebf316715b095))
* **ci:** select shards from the PR's own change; route tests and uncovered lines ([59ca51c](https://github.com/ooples/AiDotNet/commit/59ca51c33b56a5af71bb206dd9afcf8e86dc3e01))
* **ci:** select shards from the PR's own change; route tests and uncovered lines ([8401696](https://github.com/ooples/AiDotNet/commit/840169605ec55eb53d757ab66cee4040c75a6e17))
* **ci:** select shards on post-merge master pushes instead of running all 164 ([#2245](https://github.com/ooples/AiDotNet/issues/2245)) ([a25c4c9](https://github.com/ooples/AiDotNet/commit/a25c4c9faf7c74334513b4dee1cab87967ceacf4))
* **ci:** skip the shape sweeps when the worker cannot run, instead of failing ([3e6b20b](https://github.com/ooples/AiDotNet/commit/3e6b20badbf9546e2b1fbff56e538023d36c72ff))
* **ci:** stabilize attribution bindings and review fact discovery ([3cb9218](https://github.com/ooples/AiDotNet/commit/3cb92186b835be7ed2c44643f913764560798fcb))
* **ci:** stop a documentation change running the whole matrix ([ffe35f3](https://github.com/ooples/AiDotNet/commit/ffe35f3baf5721e946987cd6fb997cc8e249c540))
* **ci:** suppress unrelated validation safely ([e371fab](https://github.com/ooples/AiDotNet/commit/e371fabf099260786167259e24d53bff8e4e68f4))
* **ci:** take the Select step's context through env so master's workflow loads again ([c6462b4](https://github.com/ooples/AiDotNet/commit/c6462b46f662a734cd3dc38a02f56bc5d0cf8368))
* **ci:** take the Select step's context through env so master's workflow loads again ([19ed5a7](https://github.com/ooples/AiDotNet/commit/19ed5a7744d5d998b46f6aa4986ef6c8b1fde5f3))
* **ci:** validate selector output contracts ([c57e077](https://github.com/ooples/AiDotNet/commit/c57e07728c850eb935628c279a431c5add491382))
* claim disposal atomically and release both sequence clones ([2ba9f7c](https://github.com/ooples/AiDotNet/commit/2ba9f7cbe112dbc4bdbdb0fdba3462d78246fc5e))
* **classification:** avoid int overflow in the multi-label gradient averaging divisor ([be0fe95](https://github.com/ooples/AiDotNet/commit/be0fe951ade850b417603075e25e1d6e208edf0b))
* **clustering:** prevent overflow in ARI expected index, X-means variance and G-means correction ([c51e93f](https://github.com/ooples/AiDotNet/commit/c51e93fc318954e3577181ea70cc1c9a849390ec))
* **codeql:** filter disposal loops explicitly and log instead of swallowing cleanup failures ([bbc0943](https://github.com/ooples/AiDotNet/commit/bbc094325321213f8fc7223a5f557f861c9e8d5a))
* **codeql:** triage all 477 error alerts, fix 139 real ones, clear 2 generator false positives ([158add2](https://github.com/ooples/AiDotNet/commit/158add2e7754dd33c3d75284005a273ca90e7719))
* **codeql:** widen rank-statistic products before overflow and filter genome connections ([41457ff](https://github.com/ooples/AiDotNet/commit/41457ff81874718e137ab97a8edf7f7b12b312f3))
* collapse warmup plus cosine into one fused-expressible scheduler ([58dee39](https://github.com/ooples/AiDotNet/commit/58dee394ae8fa3549ddcc9f7be2bc00ef0e1f0b9))
* compare onnx contract shapes through toarray for net471 ([00b867c](https://github.com/ooples/AiDotNet/commit/00b867c53fb71fa916afc6a47597e8ab32ade0bd))
* **decomposition:** use a real-valued LOESS bandwidth in STL cycle-subseries smoothing ([d7fd6a1](https://github.com/ooples/AiDotNet/commit/d7fd6a1d2d5844ae4b7bbfee873f83f27712fbe8))
* **diffusion:** install the FlowMatchingScheduler linear timestep schedule instead of dropping it ([da85d4a](https://github.com/ooples/AiDotNet/commit/da85d4a70b08261b1a303ad5e5dc2c968efb31d6))
* **dimensionality-reduction:** widen n*p in the SparsePCA reconstruction error ([8b0975e](https://github.com/ooples/AiDotNet/commit/8b0975e39830c69cc05c52e20a235d4c41c2b756))
* **distributed:** keep resized layouts, hybrid topology and transfer restarts correct ([64de951](https://github.com/ooples/AiDotNet/commit/64de9517fe69ced94a8ffb0e4e9059bb3b3f9c68))
* **docs,tests:** repair ~900 uncompilable doc examples, gate them, and give the layer invariant harness subscribers ([9bcefb6](https://github.com/ooples/AiDotNet/commit/9bcefb6b9ee776d3e6e8392a2baffc9849e62ca8))
* enforce GPT4Vision ONNX graph and token contracts ([0ed2d26](https://github.com/ooples/AiDotNet/commit/0ed2d2647770733035708b167d3a28a17a72edde))
* **evaluation:** widen sample-count products in Cochran Q, DeLong, Kruskal-Wallis and Wilcoxon ([ea8f5d0](https://github.com/ooples/AiDotNet/commit/ea8f5d0bd01ed108420e7b7c829f0f11eb574c17))
* **feature-selection:** widen n*n normalisers in distance-correlation, kernel and SURF selectors ([18a1958](https://github.com/ooples/AiDotNet/commit/18a1958948817df9bd4050c5551bbf34c2996812))
* **feature-selection:** widen sample-count products in rank, contingency and point-biserial stats ([e92c72f](https://github.com/ooples/AiDotNet/commit/e92c72f85e89be534c9430bfbad1f04215f563ad))
* **finance:** add TradingEnvironment.OnReset hook so subclasses reset per-episode state ([c50f83e](https://github.com/ooples/AiDotNet/commit/c50f83ede74eb232e89a9ee43a3df39c04f08268))
* **finance:** address the seven open coderabbit threads on [#2175](https://github.com/ooples/AiDotNet/issues/2175) ([f65a14e](https://github.com/ooples/AiDotNet/commit/f65a14e94eef386d06df46ef2dd07aac77013d89))
* **finance:** bound the sac policy per the paper and repair trading option copying ([5fbb831](https://github.com/ooples/AiDotNet/commit/5fbb83145e2d675789080019f832b16b7a4e4417))
* **finance:** decay FinancialDQNAgent epsilon from EpsilonStart to EpsilonEnd per update ([3864c0a](https://github.com/ooples/AiDotNet/commit/3864c0a8252a0dfc56268f3784799c1b314e1286))
* **finance:** dispose a2c scratch tensors and make agent seeding test-order independent ([3df1155](https://github.com/ooples/AiDotNet/commit/3df1155a2780e78b1afff4e6eb465c02cc9aeffb))
* **finance:** dispose batch tensors and drop float-equality reward guard ([a441589](https://github.com/ooples/AiDotNet/commit/a441589bd55df000e6ae04d84bb4a4d05473cb2a))
* **finance:** dispose every agent PortfolioExperimentRunner creates, on all paths ([032f26c](https://github.com/ooples/AiDotNet/commit/032f26ceeaf4bc039a61021fed8d6d8552f462d5))
* **finance:** dispose the networks trading agents own ([2b66dc6](https://github.com/ooples/AiDotNet/commit/2b66dc69a1280093418e68d613ca195af50f8cd5))
* **finance:** enforce on-policy rollout ownership and shared mutation boundaries ([0eb3f08](https://github.com/ooples/AiDotNet/commit/0eb3f0834a2b3d7d5268f2abaf55d422dbc86b36))
* **finance:** financialDQNAgent never annealed its exploration rate ([695d9da](https://github.com/ooples/AiDotNet/commit/695d9daadec4a08cc66d45269f8c860892943b26))
* **finance:** FinancialDQNAgent never annealed its exploration rate ([0d084f3](https://github.com/ooples/AiDotNet/commit/0d084f3fe605dc13dd5eea287d46b662e7233b29))
* **finance:** fold [#2100](https://github.com/ooples/AiDotNet/issues/2100)'s dqn schedule fixes and learning tests into this pr ([2c78c04](https://github.com/ooples/AiDotNet/commit/2c78c04f273384e1ff0d13511a3c45e042b92d74))
* **finance:** honour TradingAgentOptions WarmupSteps and HiddenLayers in trading agents ([7a3de9c](https://github.com/ooples/AiDotNet/commit/7a3de9cc2700ef9b786287777beaecfa16e9e49e))
* **finance:** implement the SAC, market-making and A2C learning updates ([492d92f](https://github.com/ooples/AiDotNet/commit/492d92f7f99d5524b74c4fb8ca77283570b62cc0))
* **finance:** implement the SAC, market-making and A2C learning updates ([2bfa682](https://github.com/ooples/AiDotNet/commit/2bfa6827e9d6b230523b8e1b7c5150773558f5c7))
* **finance:** make FinancialA2CAgent a softmax policy trained by advantage policy gradient ([e2fd3b5](https://github.com/ooples/AiDotNet/commit/e2fd3b5438482229c6a74a5ab4cd80d87b764026))
* **finance:** make RecurrentPolicyAgent release the LSTM cell it owns ([9d5850d](https://github.com/ooples/AiDotNet/commit/9d5850da3a7aa0fc3a8818f14e20b87e071db103))
* **finance:** make seeded financial agents train bit-identically (target sync, weight init) ([5134269](https://github.com/ooples/AiDotNet/commit/513426930aeabf17c7eb9e83cda97f7b86e34d1c))
* **finance:** repair exploration, policy gradient, determinism and dead options in financial RL agents ([f0244a6](https://github.com/ooples/AiDotNet/commit/f0244a66282092e8fc404c0b30638a00d8ed4fc0))
* **finance:** trim the incomplete window in N-HiTS tape pooling instead of throwing ([f94efeb](https://github.com/ooples/AiDotNet/commit/f94efeb41c00eb6e1079ed6d71b5464b12257aa6))
* **finance:** use zero-mean seeded Gaussian exploration noise in SAC and market-making agents ([f5b20a1](https://github.com/ooples/AiDotNet/commit/f5b20a18d0b5568c7e433ce7f7b390460785918e))
* **finch:** train through the published finch hyperparameters ([f2b67e9](https://github.com/ooples/AiDotNet/commit/f2b67e9b33b86feee6d4817511b5e140cb18b0cd))
* **gan:** keep wgan's critic score on the tape and close the aidn101 ratchet ([81832ea](https://github.com/ooples/AiDotNet/commit/81832ea15a14837050e86700be3bdfab12538e78))
* **generators:** give audiopalm the 15-step memorization window ([3e31b04](https://github.com/ooples/AiDotNet/commit/3e31b04b072c5eeb97b66a3b73f22700b40a4ecb))
* **generators:** leave registered parameter components to the registry ([fecac2e](https://github.com/ooples/AiDotNet/commit/fecac2e67949dce72cee8cf4e29affc60ae6bcbb))
* **generators:** make containing-type walks provably non-null for CodeQL ([eebe877](https://github.com/ooples/AiDotNet/commit/eebe877796dcdbf4867277b4c2e41bd98b80f56f))
* give generated AudioPaLM probe sufficient paper-optimizer steps ([9f7dc4b](https://github.com/ooples/AiDotNet/commit/9f7dc4bec887136ea943e15a4b3ac8635c86fef4))
* **helpers:** rethrow a lone disposal failure unchanged; aggregate only several ([202e60b](https://github.com/ooples/AiDotNet/commit/202e60b4b0abe59e5fc2dd29dbf8ee33958e215e))
* in-place parameter updates ([#1842](https://github.com/ooples/AiDotNet/issues/1842)) + CI cannot report a dead shard as success ([#2086](https://github.com/ooples/AiDotNet/issues/2086)) ([3185b41](https://github.com/ooples/AiDotNet/commit/3185b41f1e1cffb81d76130e319233153c984471))
* isolate distributed test sessions and synchronize dashboard readers ([d0f9777](https://github.com/ooples/AiDotNet/commit/d0f9777ba7271dee06fe6e0e97c97ae2269f04f8))
* keep Autoformer gradient accumulators alive across tape resets ([588c764](https://github.com/ooples/AiDotNet/commit/588c76454c709d347bd0c6ac507beddf4bd79e1c))
* **kernels:** widen count products in spectrum and bag-of-words string kernels ([fb9a75c](https://github.com/ooples/AiDotNet/commit/fb9a75c9e4cad6acf7d7be7a9a45984dfeb0916b))
* make disposing models and RL agents actually release what they own ([e1ae8af](https://github.com/ooples/AiDotNet/commit/e1ae8afd6c8c627ae74e041e78d2c7b7cb887cf9))
* measure reward following differentially in the rl policy invariant ([7bdb525](https://github.com/ooples/AiDotNet/commit/7bdb5252bc7d258905c94c731e5b7c533dbe32da))
* measure reward following differentially in the rl policy invariant ([c949190](https://github.com/ooples/AiDotNet/commit/c9491904bb0c241fc0254ebc856c096d050b0bac))
* measure the reward-following probe along the axis that separates the actions ([4ab7ba4](https://github.com/ooples/AiDotNet/commit/4ab7ba488fdbce6ffc0dbaac63a3482134d0e9db))
* **meta-learning:** widen the parameter index product in BOIL and Matching Networks ([08b5e7d](https://github.com/ooples/AiDotNet/commit/08b5e7d37e31d81c60d49ca1a79295751ee1a238))
* **metadata:** stop model metadata from serializing the model eagerly ([#1830](https://github.com/ooples/AiDotNet/issues/1830)) ([c32502a](https://github.com/ooples/AiDotNet/commit/c32502a70112f7d09851107088e01f3744b89a70))
* **models:** dispose the model copies AiModelResult makes for inference and ensembles ([557c719](https://github.com/ooples/AiDotNet/commit/557c71976ceba5653f06ac29397bb0d6e1de03fd))
* **models:** release the meta-learning result's separate best solution exactly once ([0270f36](https://github.com/ooples/AiDotNet/commit/0270f36357626e9d138cf1bf42e237c8972c780f))
* **models:** repair the three shard failures in batch 6 ([dee101e](https://github.com/ooples/AiDotNet/commit/dee101e318ba01cc994a65668d861f765d6fc0fb))
* **neat:** apply sigmoid node activations in Genome.Activate ([1f9ad8e](https://github.com/ooples/AiDotNet/commit/1f9ad8ebe407ae6efeec90a7a3efd06af16ca88e))
* **neural-networks:** refuse a second model on one architecture instead of aliasing it ([c3d0b84](https://github.com/ooples/AiDotNet/commit/c3d0b84d000b002f1cef633e6746249bf2dbf8a6))
* **neural-networks:** widen element-count products before they overflow int32 ([7094add](https://github.com/ooples/AiDotNet/commit/7094add36616bdf48b98d696ef7eaf5bbb39f15b))
* **nn:** make a clone serialize like its source and let ResolveShapes run the real forward ([843d677](https://github.com/ooples/AiDotNet/commit/843d67722348b512acf65e474413b8dc47d261d5))
* **nn:** make NeuralNetworkBase.Dispose idempotent and guard repeated model disposes ([ac580d3](https://github.com/ooples/AiDotNet/commit/ac580d34480e987fd9b7c84004b726b5f71b9281))
* **nn:** stop a repeated layer Dispose from re-running derived teardown ([a415224](https://github.com/ooples/AiDotNet/commit/a415224a03f194cee29e1162e811c14eb079b864))
* **onnx:** enforce ImageBind and LLaVA graph contracts ([02d5315](https://github.com/ooples/AiDotNet/commit/02d5315e87a32d2cd73c01392e644c9ef54783a3))
* **optimizers:** step declared schedules during training instead of freezing them ([e5214a7](https://github.com/ooples/AiDotNet/commit/e5214a7653d7b3b7cdddf8ba6d0a2d6377cd63a9)), closes [#1928](https://github.com/ooples/AiDotNet/issues/1928)
* **options:** keep constructor validation assembly-internal ([bb44595](https://github.com/ooples/AiDotNet/commit/bb44595e5174cd2f7bf48689230a25eb271a8ff8))
* **options:** preserve sequence copies and validate consumed settings ([80e7540](https://github.com/ooples/AiDotNet/commit/80e7540d69806c5e26de7a38b12c98932604d979))
* **options:** preserve sequence copies and validate consumed settings ([f8f6488](https://github.com/ooples/AiDotNet/commit/f8f648811cc5cf722630ae03461371cae1632cce))
* **options:** remove unconsumed shared gradient alias ([05f58b9](https://github.com/ooples/AiDotNet/commit/05f58b9e4e83dc9796f7dd02a7590842cdeac2ad))
* **options:** restore internal Validate and Clip's path-first order ([d32c5c2](https://github.com/ooples/AiDotNet/commit/d32c5c27a6346202295621f9ada0a257cb1dfcb2))
* **options:** restore the rest of what the master merge dropped ([4e5958b](https://github.com/ooples/AiDotNet/commit/4e5958ba8f62cc286d71dac34558f8b6209eedd3))
* **options:** review fixes for [#2130](https://github.com/ooples/AiDotNet/issues/2130) ([24e82fd](https://github.com/ooples/AiDotNet/commit/24e82fd33538230374276841ef903129696872bc))
* **options:** separate ONNX-only input configuration from native towers ([9511008](https://github.com/ooples/AiDotNet/commit/95110084d52c88da3d3189cdd7b91e9a9802ad34))
* **options:** use canonical GPT4 native vision width ([760f2f9](https://github.com/ooples/AiDotNet/commit/760f2f9138d5f3710248f29596d6735bcd4a564b))
* **options:** validate shared GAN channels and clarify migrated VLM documentation ([33ed850](https://github.com/ooples/AiDotNet/commit/33ed850d4845892c884c929b437b38b82ebcc6dd))
* **pinn:** average the multi-scale PINN epoch loss over the batches actually processed ([64a39c6](https://github.com/ooples/AiDotNet/commit/64a39c6a8e756bd8c189fffec86b211c81ddc536))
* **playground:** reference AiDotNet by project instead of the 0.113.0 package ([bf05125](https://github.com/ooples/AiDotNet/commit/bf05125b5f1c05ffa6977683f1f4ad8312f42762))
* **playground:** stop logging email-derived data in the stripe webhook ([bfc414f](https://github.com/ooples/AiDotNet/commit/bfc414fc09fd03597502d8cce27ace09a6e843f9))
* **preprocessing:** prevent int32 overflow in Otsu histogram sums and between-class variance ([c2b4ce6](https://github.com/ooples/AiDotNet/commit/c2b4ce6634fb7ec591fca31c4e58cdf72204cf8d))
* repair facade shard failures through shared cloning and distributed paths ([282ffc1](https://github.com/ooples/AiDotNet/commit/282ffc15ee69c3b98bf0017dbcb6443d24c22e5e))
* repair the tests that had never actually run ([af0ec4e](https://github.com/ooples/AiDotNet/commit/af0ec4ebae820d6c45325dfd39df4ce754ac47e0))
* require state-bound on-policy action provenance ([77fbd42](https://github.com/ooples/AiDotNet/commit/77fbd427f7981a6aee818927a1123c41f9dd7f69))
* **review:** verify sequence scaffold rules and options documentation ([4713636](https://github.com/ooples/AiDotNet/commit/47136363b08ac1eabc39c63b371bf6b125c6f0c5))
* **review:** verify sequence scaffold rules and options documentation ([19c637e](https://github.com/ooples/AiDotNet/commit/19c637e6c9220858831d879a44189ed122fecd6b))
* **rl,gan:** match the sac objective to its behaviour policy and untangle the gan concat ([331f596](https://github.com/ooples/AiDotNet/commit/331f5964291dc60d81ae4a61c0d4fbd4e12e831e))
* **rl:** carry the legal-action mask into the PPO update, not just the selection ([cb9601d](https://github.com/ooples/AiDotNet/commit/cb9601dcdaf8e3037dbc0dd3ec461456f3998c8d))
* **rl:** close the five review findings on legal-action masking ([6df3bbc](https://github.com/ooples/AiDotNet/commit/6df3bbc7e05c8689c986c3d2dce7deb6127594cd))
* **rl:** register every deep agent network so disposing the agent releases it ([5a44106](https://github.com/ooples/AiDotNet/commit/5a441065115a51d307290bd54f9241bdcf1d1188))
* **rl:** release every agent network even when one network's Dispose throws ([41abacd](https://github.com/ooples/AiDotNet/commit/41abacdc5755c5b0d954b50c3e472d9c9294d965))
* **rl:** restore the gradient path in the sac actor update ([c3d9e99](https://github.com/ooples/AiDotNet/commit/c3d9e994db49db59eadc7b27aa5179fa64861890))
* **rl:** retain action legality through A2C updates and DQN replay targets ([c38b49d](https://github.com/ooples/AiDotNet/commit/c38b49d70688a443e1f25b23254acab83cf99352))
* **rl:** size rl fixtures from the model and repair dreamer's dead actor ([ca748d6](https://github.com/ooples/AiDotNet/commit/ca748d6811fa140580ea40de0517e6f3361d575b))
* **runtime:** invalidate resized gradients and isolate clone shapes ([d3ef93c](https://github.com/ooples/AiDotNet/commit/d3ef93ceff4e63d79aae3a10eb0e22f3af07f923))
* **runtime:** restore layer shapes and refresh lazy distributed layouts ([46e7489](https://github.com/ooples/AiDotNet/commit/46e748999baeb807747cfca86a299266f994f133))
* **runtime:** restore layer shapes and refresh lazy distributed layouts ([2dece39](https://github.com/ooples/AiDotNet/commit/2dece39bcfc6ddb2702cc0930bb478a6e88d2fe2))
* scope the target-dependence wall-clock budget to the probe, not to one call ([1e7cc6f](https://github.com/ooples/AiDotNet/commit/1e7cc6f9ab4760fc1106276ca989422c887753b2))
* **serving:** close SQLite sandbox escape and neutralize logged request data ([c332e2a](https://github.com/ooples/AiDotNet/commit/c332e2a7e60fa7e71bb30476241e04167ddc2ea4))
* **serving:** neutralize caller-controlled values before logging them ([f47ffbc](https://github.com/ooples/AiDotNet/commit/f47ffbc27e672e8909063520857f90a7244665dc))
* **serving:** stop sandboxed sqlite from attaching host database files ([94e417e](https://github.com/ooples/AiDotNet/commit/94e417ec95022a200cebc61db73438c90a709123))
* **statistics:** widen pair-count products in Kendall tau, KID and the MMD domain adapter ([dd0a38f](https://github.com/ooples/AiDotNet/commit/dd0a38f37fd549f8b98f0b7d0108467a9c9e33ab))
* stop monte carlo es predict from cancelling the exploring start ([b94fe7d](https://github.com/ooples/AiDotNet/commit/b94fe7d3862e3e35786bf2302c0e9058c3763a89))
* stop the clone bit-identity check from running between the two forwards ([2dd2ee1](https://github.com/ooples/AiDotNet/commit/2dd2ee151254c303020fb71e013d374725134776))
* **survival:** compute the random survival forest log-rank variance in double ([78f0feb](https://github.com/ooples/AiDotNet/commit/78f0feb55d54631bb69a886ed7375c810e5b9dc5))
* **test-impact:** typed failures for null bundle members and an unbounded TRX ([2bc8c1d](https://github.com/ooples/AiDotNet/commit/2bc8c1daac621cf3ef6b9a7c981d22a1fd537fed))
* **testing:** lower rl warmup gates instead of outlasting them ([1b7c8a7](https://github.com/ooples/AiDotNet/commit/1b7c8a70f85328447fd968c1befc9368e2172e70))
* **testing:** size rl fixtures from the model, not a hardcoded state width ([3333074](https://github.com/ooples/AiDotNet/commit/333307490464c6a20e00e790782ae0d085bcda4e))
* **tests:** clear the warmup gate in the dqn exploration schedule fixture ([491b58f](https://github.com/ooples/AiDotNet/commit/491b58fb89e48a312bf8ae73088ac48e1c491184))
* **tests:** tell an initialisation gap from a training defect in the train/test invariant ([08b860d](https://github.com/ooples/AiDotNet/commit/08b860d6ef524c86e9a0ecd6438839b9c3ba0954))
* **tests:** use APIs that exist on net471 in the paper-optimizer tests ([c9deb46](https://github.com/ooples/AiDotNet/commit/c9deb4655e701a788ce4cf214067e6ae67a70e94))
* **tests:** use portable typed shape assertions in audio probes ([d45d93e](https://github.com/ooples/AiDotNet/commit/d45d93ebbeae755e16486de0420429d4e3cd76e5))
* **tests:** validate clone fixture with the real generator pipeline ([fb27f0a](https://github.com/ooples/AiDotNet/commit/fb27f0aa4e08c9d0bdf397286b26de2cb196fb50))
* **timeseries:** preserve Autoformer gradients across arena resets ([2f0a699](https://github.com/ooples/AiDotNet/commit/2f0a699b416159261816bbf0f546670305ed0fd7))
* **tts:** rebuild a cloned Bark around its own codec, not a fresh default one ([2860c39](https://github.com/ooples/AiDotNet/commit/2860c398806f5b29ae0ee0528b505593c12188e6))
* validate multimodal contracts and restore shared model state ([6b1f3ca](https://github.com/ooples/AiDotNet/commit/6b1f3ca7da6c95f7fde1cd6534e28175224d2f93))
* validate visual-query ONNX configuration against loaded graphs ([55e9e8d](https://github.com/ooples/AiDotNet/commit/55e9e8d8e951ebf99b1952dd1c19d595bc977d60))
* **video:** declare VideoMAE's default input as a clip it can embed ([44208d2](https://github.com/ooples/AiDotNet/commit/44208d2ca4c7b3cb5a16b6c097c791a145548b83))
* **vislang:** close the onnx and correspondence review findings ([06831e6](https://github.com/ooples/AiDotNet/commit/06831e63ac7e0c69f15d31f96f346811be6164bc))
* **vislang:** validate consumed native geometry and preserve valid defaults ([5a5f5b0](https://github.com/ooples/AiDotNet/commit/5a5f5b0600e6b2b77036ded95e345d24d35c08ca))
* **wavelets:** use floating-point 3/2 in the Meyer wavelet second transition band ([721c7a6](https://github.com/ooples/AiDotNet/commit/721c7a62e329f59fa35a15435262bdc386ad4694))


### Refactoring

* drop an unused local from the rainbow update ([db4df03](https://github.com/ooples/AiDotNet/commit/db4df0341a6ab8b1f401d69fd2edb16f5768f143))
* remove write-only collections flagged by CodeQL cs/unused-collection ([0951d2d](https://github.com/ooples/AiDotNet/commit/0951d2d9339eb7367671c9f6aa9c963202b3e557))


### Build System

* tests/AiDotNet.Tests -f net10.0, 0 errors. ([4ec42f0](https://github.com/ooples/AiDotNet/commit/4ec42f0ed2d73237238a84e884135d5f35d0931e))


### Documentation

* **#2090:** spec — model Options as the user-facing configuration surface ([c2d99cf](https://github.com/ooples/AiDotNet/commit/c2d99cf60118fd1310be09e82f50e51197eae62b))
* **ci:** correct delta validation planning contract ([45ac559](https://github.com/ooples/AiDotNet/commit/45ac55994ffd480b71e66857a61a06ff524979fa))
* **ci:** record independent review and delta reuse proofs ([133f89b](https://github.com/ooples/AiDotNet/commit/133f89b720f31b8fbec99f97b15e9bac1c728209))
* **ci:** record routing canary evidence ([88364e9](https://github.com/ooples/AiDotNet/commit/88364e91fc17f0562440415859542abca579c74c))
* **ci:** record routing canary evidence ([bcf0aaf](https://github.com/ooples/AiDotNet/commit/bcf0aaf5b0f187305a45b514e29bd9964184f8a7))
* **ci:** record startup evidence and native proof boundary ([9de7c98](https://github.com/ooples/AiDotNet/commit/9de7c98b20b5b5c24c2229349bd3726819e88364))
* correct the placeholder-defaults claim, measured on the wrong branch ([cea3a45](https://github.com/ooples/AiDotNet/commit/cea3a4578e3c14a23b90f4ea3d68b74cb473a36b)), closes [#2090](https://github.com/ooples/AiDotNet/issues/2090)
* **options:** reconcile typed configuration contracts and complete phase accounting ([e2df961](https://github.com/ooples/AiDotNet/commit/e2df9612d7eb6bb2c87b13be4683aa7f38e10e06))
* pass eagle options to the eagle language model example ([9d5dfc5](https://github.com/ooples/AiDotNet/commit/9d5dfc5d4bdfdf9e97797e3c7e651dc354d4898f))
* prove audio regressions on every test framework ([7f5f6cf](https://github.com/ooples/AiDotNet/commit/7f5f6cfa4f808c5762d8ac71b522c5177722dd7e))
* record failure-first action provenance verification ([480be82](https://github.com/ooples/AiDotNet/commit/480be824b07e4eceda3ce8cf328a49c44f51d540))
* record PR2130 native options validation proof ([731bcd3](https://github.com/ooples/AiDotNet/commit/731bcd30bef6290dee2e5dd9a438d7bcd1f41158))
* **review:** record 419 integrated test passes and binary hashes ([34ac3a7](https://github.com/ooples/AiDotNet/commit/34ac3a71e6909d61ced7325a801b99d97b78ede8))
* **review:** record 419 integrated test passes and binary hashes ([31ad157](https://github.com/ooples/AiDotNet/commit/31ad157f2a657a7f29926168b6a93b20531d2759))
* **review:** record independent actual-model options proof ([855e0d5](https://github.com/ooples/AiDotNet/commit/855e0d5093673cc5d74af36f24f029b855a5890b))
* **review:** record independent actual-model options proof ([dcafab9](https://github.com/ooples/AiDotNet/commit/dcafab97f30b0589a65d9b0e3097f3d8289e4434))
* **vision-mamba:** pass the example's sizes through VisionMambaOptions ([d57f0bb](https://github.com/ooples/AiDotNet/commit/d57f0bbb6f1511a005587466867ddfd34206c036))

## [Unreleased]

### Breaking Changes

* Sequence language-model dimensions now belong on their model-specific options objects. In particular, `FalconMambaLanguageModel<T>` takes `FalconMambaOptions` as its second constructor argument, before `ILossFunction<T>`. Replace positional calls such as `new FalconMambaLanguageModel<float>(architecture, loss)` with `new FalconMambaLanguageModel<float>(architecture, lossFunction: loss)`, and move scalar dimensions into `options: new FalconMambaOptions { ... }`.
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
