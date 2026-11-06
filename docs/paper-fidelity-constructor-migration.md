# Constructor migration

These constructor changes are recorded in `src/PublicAPI.Unshipped.txt` and
require recompiling consumers against the new package.

For DGCNN, replace `new DGCNN<float>(numClasses, channels, loss)` with
`new DGCNN<float>(new DGCNNOptions { NumClasses = numClasses, EdgeConvChannels = channels }, loss)`.
The former three-argument options constructor is replaced by
`new DGCNN<float>(options, loss)`. Consolidate model settings into that single
`DGCNNOptions` instance; the separate `modelOptions` parameter was removed.

`new PrioritizedReplayBuffer<float>(capacity)` remains valid source code, but its
constructor now takes an optional `int? seed`. Recompile existing binaries.
Pass a seed for reproducible sampling, or leave it null for secure entropy.

Package validation retains exact baseline exceptions for these three removed
signatures on net10.0, net8.0 and net471. Other compatibility checks remain enabled.
