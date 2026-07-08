from train_imagenet1k_timm_baseline_common import main_for_model


MODEL_SPEC = {
    "display_name": "Swin-Tiny",
    "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
    "search_pattern": "*swin_tiny*",
    "candidates": [
        "swin_tiny_patch4_window7_224.ms_in1k",
        "swin_tiny_patch4_window7_224.in1k",
        "swin_tiny_patch4_window7_224",
    ],
    "checkpoint_dir": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\swin_tiny",
}


if __name__ == "__main__":
    main_for_model(MODEL_SPEC)
