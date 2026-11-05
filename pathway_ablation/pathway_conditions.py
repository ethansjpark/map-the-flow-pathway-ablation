"""Layer windows for the 32-layer LLaVA-NeXT-7B pathway ablation."""

INTER_FRAME = list(range(5, 15))
VIDEO_TO_QUESTION = list(range(5, 20))
QUESTION_TO_LAST = list(range(15, 25))

# Each tuple is (inter-frame, video-to-question, question-to-last).
# Full uses the baseline only; random uses edge sampling instead of windows.
CONDITIONS = {
    "full": None,
    "effective": (INTER_FRAME, VIDEO_TO_QUESTION, QUESTION_TO_LAST),
    "random": None,
    "shift_minus3": (list(range(2, 12)), list(range(2, 17)), list(range(12, 22))),
    "shift_plus3": (list(range(8, 18)), list(range(8, 23)), list(range(18, 28))),
    "shrink": (list(range(7, 13)), list(range(7, 18)), list(range(17, 23))),
    "drop_inter": ([], VIDEO_TO_QUESTION, QUESTION_TO_LAST),
    "drop_v2q": (INTER_FRAME, [], QUESTION_TO_LAST),
    "drop_q2l": (INTER_FRAME, VIDEO_TO_QUESTION, []),
}


def build_block_layer_lists(windows, num_layers=32):
    """Return layers where each attention edge group is blocked."""
    inter_frame, video_to_question, question_to_last = windows
    return {
        "vision_inter_frame": [i for i in range(num_layers) if i not in inter_frame],
        "vision_to_question": [i for i in range(num_layers) if i not in video_to_question],
        "question_to_last": [i for i in range(num_layers) if i not in question_to_last],
        "last_to_last": list(range(num_layers)),
        "vision_to_last": list(range(num_layers)),
        "vision_to_vision": list(range(20, num_layers)),
        "question_to_question": list(range(25, num_layers)),
    }
