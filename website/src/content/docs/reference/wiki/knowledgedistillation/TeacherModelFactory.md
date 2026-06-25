---
title: "TeacherModelFactory<T>"
description: "Factory for creating teacher models from enums and configurations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.KnowledgeDistillation`

Factory for creating teacher models from enums and configurations.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAdaptiveTeacher(IFullModel<,Vector<>,Vector<>>,Nullable<Int32>)` | Creates an adaptive teacher model wrapper. |
| `CreateCurriculumTeacher(IFullModel<,Vector<>,Vector<>>,Nullable<Int32>,CurriculumStrategy)` | Creates a curriculum teacher model wrapper. |
| `CreateTeacher(TeacherModelType,IFullModel<,Vector<>,Vector<>>,ITeacherModel<Vector<>,Vector<>>[],Double[],Nullable<Int32>,OnlineUpdateMode,Double,CurriculumStrategy,Int32,AggregationMode)` | Creates a teacher model from the specified type and configuration. |

