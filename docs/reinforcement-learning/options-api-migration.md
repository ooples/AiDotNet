# RL options API migration (#2414)

The RL rebuild intentionally changes the released option contracts. The removed
members and constructors are recorded in `src/PublicAPI.Unshipped.txt`.

| Previous API | Current API and reason |
| --- | --- |
| Agent `Optimizer` properties and vector-optimizer constructor arguments | Configure `OptimizerFactory` with tensor optimizers. The previous vector optimizers could not train the agents' tensor networks. The factory creates independent optimizer state for each network. |
| A2C, A3C and PPO `ValueLossCoefficient` | Configure the policy and value learning rates. The coefficient was not applied by these implementations. |
| CQL and SAC generic `TargetEntropy` | Use `double? TargetEntropy`. Unconstrained `T?` did not represent an optional numeric value; `null` now selects the action-size heuristic and zero remains a real target. |
| Dreamer `LatentSize` | Configure `DeterministicSize` and `StochasticSize` for the RSSM's two distinct state components. |
| MuZero `UseValuePrefix` | The model implements MuZero's value prediction; the unused EfficientZero-style switch is removed. |
| Three-argument `DuelingCombinationLayer` constructor | Use named `seed:`; the extended constructor also configures categorical atoms and the value/advantage stream widths. |
| Dreamer's generated parameter-registrar interface | Dreamer registers its complete native network inventory through `RegisterComponents`. Use its model parameter accessors rather than the former generated registration interface. |

`CompatibilitySuppressions.xml` records only these exact released-baseline
differences for `net10.0`, `net8.0` and `net471`. The entries do not suppress
cross-framework consistency checks or disable package validation.

The failing CI job reported 40 distinct removed-member diagnostics and one
Dreamer registration-interface diagnostic on each of the three frameworks.
The SDK generated the exact symbol identifiers. Each removed member was checked
against the existing public API removal ledger before its exception was added.

Local validation uses the unchanged net10.0 assembly from CI run 41532056553,
compared with the released 0.234.0 package: the original exceptions fail, the
updated exceptions pass, and deleting one new exception restores the corresponding
CP0002 failure. The full multi-framework package build still runs in hosted CI.
