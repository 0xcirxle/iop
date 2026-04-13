"""Static label metadata for the saved models."""

TASK_LABELS = {
    "binary": {0: "Healthy", 1: "Faulty"},
    "severity": {0: "Healthy", 1: "1u", 2: "3u", 3: "5u"},
    "phase": {0: "Healthy", 1: "Phase 1", 2: "Phase 2", 3: "Phase 3"},
    "load": {0: "No Load", 1: "Half Load", 2: "Full Load"},
}

TASK_ORDER = ("binary", "severity", "phase", "load")


def get_task_labels(binary_labels_swapped: bool = False):
    if not binary_labels_swapped:
        return TASK_LABELS

    swapped = dict(TASK_LABELS)
    swapped["binary"] = {
        0: TASK_LABELS["binary"][1],
        1: TASK_LABELS["binary"][0],
    }
    return swapped
