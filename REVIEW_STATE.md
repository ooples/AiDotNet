
## #2032 conflict resolution (in progress)
Branch fix2032 = origin/fix/training-step-target-independence + merge origin/master.
RESOLVED:
 - Directory.Packages.props: kept master's 0.129.2 (all 4 packages, lockstep) AND
   preserved #2032's rationale comment, rewritten to describe 0.129.2. XML validated
   (NOTE: "--" is ILLEGAL inside an XML comment; it silently invalidates the whole
   props file and NU1015 then fires on unrelated packages).
 - NeuralNetworkModelTestBase.cs: kept #2032's doc comment (master side empty; the
   MakeTargetWellPosedForLoss call it documents exists on both sides).
OPEN / decision point:
 - DeepBeliefNetworkTests.cs: BOTH sides changed it.
   master(#2026) refined the CD-1 pre-training overrides (MoreData compares against
   its own post-CD baseline). #2032 DELETED all overrides (209 lines) because it fixed
   the product: "RBMLayer batched pretraining no longer flattens the visible axis;
   Bernoulli visible data normalized safely; CD mutates registered parameter tensors
   in place". PLAN: take #2032's deletion, then EMPIRICALLY run the DBN tests on the
   merged tree. If they pass without overrides, deletion is correct. If they fail,
   restore master's refined overrides instead.
