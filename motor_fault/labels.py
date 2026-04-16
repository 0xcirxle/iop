"""Static label metadata for the saved models."""

TASK_LABELS = {
    "binary": {0: "Healthy", 1: "Faulty"},
    "severity": {0: "1u", 1: "3u", 2: "5u"},
    "phase": {0: "Phase 1", 1: "Phase 2", 2: "Phase 3"},
    "load": {0: "No Load", 1: "Half Load", 2: "Full Load"},
}

TASK_ORDER = ("binary", "severity", "phase", "load")
