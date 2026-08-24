# Empty-set convention

- GT ADD empty and prediction ADD empty → precision=recall=F1=1.0 for that side.
- Same for REMOVE.
- Combined delta pools ADD and REMOVE TP/FP/FN; if all zero → combined F1=1.0.
- One side empty and the other non-empty → standard precision/recall (empty prediction with non-empty GT → recall=0).
- Add Transition-Macro Dyn-F1 averages **combined** transition F1 over add_object transitions (not averaging added.f1=0 on structurally irrelevant ADD sides of remove/transfer).
- Transition-Macro Dyn-F1 must not be called micro-F1.
