"""Compare paper windows to the original 7B block-layer expressions."""

from pathway_conditions import CONDITIONS, build_block_layer_lists


def main():
    num_layers = 32
    inter_frame_range = list(range(5, 15))
    video_to_question_range = list(range(5, 20))
    question_to_last_range = list(range(15, 25))

    # Original effective-pathway-7b expressions, without token key/query ranges.
    original = {
        "vision_inter_frame": [i for i in range(num_layers) if i not in inter_frame_range],
        "vision_to_question": [i for i in range(num_layers) if i not in video_to_question_range],
        "question_to_last": [i for i in range(num_layers) if i not in question_to_last_range],
        "last_to_last": [i for i in range(num_layers)],
        "vision_to_last": [i for i in range(num_layers)],
        "vision_to_vision": [i for i in range(video_to_question_range[-1] + 1, num_layers)],
        "question_to_question": [i for i in range(question_to_last_range[-1] + 1, num_layers)],
    }
    actual = build_block_layer_lists(CONDITIONS["effective"], num_layers)
    assert actual.keys() == original.keys()
    for block_name in original:
        assert actual[block_name] == original[block_name], block_name

    for name, windows in CONDITIONS.items():
        if windows is None:
            print(f"{name}: INTER=N/A V2Q=N/A Q2L=N/A")
            continue
        assert all(0 <= layer < num_layers for window in windows for layer in window), name
        print(f"{name}: INTER={len(windows[0])} V2Q={len(windows[1])} Q2L={len(windows[2])}")
    print("PASS: effective layer lists match all seven original blocks")


if __name__ == "__main__":
    main()
