from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

# Register ucr_wblm_moby_history config under NEW_EMBODIMENT
register_modality_config(
    MODALITY_CONFIGS["ucr_wblm_moby_history"],
    EmbodimentTag.NEW_EMBODIMENT
)
