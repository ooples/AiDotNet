# Hosted validation follow-up

At consumer67fe49747, hosted compiler and consumer tests passed on net8 and net10. The net10 noise-study verifier then rejected stdout as invalid JSON (`Unexpected character ... O`, run34992943361). A dedicated `--output` stream now isolates evidence from dependency diagnostics; no text stripping or verification weakening. Existing report paths fail before any measurement, and invalid arguments fail closed.

Targeted final build: 0 warnings/errors. Dedicated-file run: all18 fixed runs valid,5768 fresh fits,1680 timing invocations,6608 correctness checks,14056 charged calls. All6 corrupted reports rejected. Overwrite and invalid-argument checks passed. This is a protocol regression rerun, not a new independent statistical confirmation or a retuning of the registered US-08 study. Original evidence archives remain unchanged.

Ordinary package-path website, documentation and census jobs still require release of the matching Evolution APIs; pinned-source validation does not remove that gate. Core PR51 current head de278ed has all hosted checks green. User handles dependency publication, reviews and merges.
