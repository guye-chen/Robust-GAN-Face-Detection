from train_imagenet1k_timm_baseline_common import main_for_model


MODEL_SPEC = {
    "display_name": "ConvNeXt-Tiny",
    "model_name": "convnext_tiny.fb_in1k",
    "search_pattern": "*convnext_tiny*",
    "candidates": [
        "convnext_tiny.fb_in1k",
        "convnext_tiny.in1k",
        "convnext_tiny",
    ],
    "checkpoint_dir": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\convnext_tiny",
}


if __name__ == "__main__":
    main_for_model(MODEL_SPEC)
