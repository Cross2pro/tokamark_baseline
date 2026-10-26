import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp

from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports
# ----------------------------------------------------------------------------------------------------------------------

try:
    from globals import REPO_ROOT
except ImportError:
    from .globals import REPO_ROOT

from tokamark.tools.utils import get_device
from tokamark.data_split import get_train_test_val_shots
from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.tools.transforms.compose_transform import (
    ComposeTransforms,
)
from tokamark.data import (
    initialize_MAST_dataset, 
    initialize_TokaMark_dataset,
)

from tokamark.evaluator import (
    WindowMetricsAccumulator,
    compute_metrics,
)


from src.multi_conv_mlp_model import (
    create_cnn_architecture
)
from src.multi_conv_lstm_model import (
    create_lstm_architecture
)
from src.multi_conv_tcn_model import (
    create_tcn_architecture
)
from src.model_transform import (
    ModelTransform_1,
    ModelTransform_2,
)
from src.trainer import (
    model_collate_fn,
)

from src.evaluator import (
    cnn_unstd_evaluation_per_shot,
)

from tokamark.tools.path import (
    RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE, 
    RANDOM_SPLIT_SIGNALS_STATS_FILE,
    TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE, 
    TEMPORAL_SPLIT_SIGNALS_STATS_FILE
    )

from MAST_tools.utils.path_utils import (
    RANDOM_SPLIT_OUTLIER_METADATA_FILE,
    TEMPORAL_SPLIT_OUTLIER_METADATA_FILE,
    )


# ----------------------------------------------------------------------------------------------------------------------

# Set device
device = get_device()
# print(f"Using device: {device}\n")


# ======================================================================================================================
if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="task_1-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/src/config/config_model_test.yaml",
        help="Path to the model YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Specified seed"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "lstm", "tcn"],
        default="cnn",
        help="Model type to train."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["random", "temporal"],
        default="random",
        help="Splitting used."
    )
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Some configuration tasks
    # ------------------------------------------------------------------------------------------------------------------

    # Load Task YAML config
    config_task = get_task_config(task_name=args.task)

    # Load CNN YAML config
    with open(REPO_ROOT + args.config, "r") as f:
        config = yaml.safe_load(f)
    
    SEED = args.seed
    print(SEED)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------

    dict_task_metadata = get_task_metadata(
        config_task=config_task,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Load correct settings
    # ------------------------------------------------------------------------------------------------------------------

    if args.split == "random":

        DATA_SPLIT = RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
        OUTLIER_FILE = RANDOM_SPLIT_OUTLIER_METADATA_FILE
        SIGNAL_STATS = RANDOM_SPLIT_SIGNALS_STATS_FILE

    elif args.split == "temporal":

        DATA_SPLIT = TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE
        OUTLIER_FILE = TEMPORAL_SPLIT_OUTLIER_METADATA_FILE
        SIGNAL_STATS = TEMPORAL_SPLIT_SIGNALS_STATS_FILE
    
    else:
        print("Unknown split.")
        raise ValueError("Unknown split.")

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------------------------------------------------------

    if config.get("data_splits_file"):
        DATA_SPLIT = config["data_splits_file"]
        print(f"[INFO] Using custom data_splits_file: {DATA_SPLIT}")

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config["subset_of_shots"],
        shuffle=config["shuffle"],
        data_splits_file_path = DATA_SPLIT        
    )
    print(f"[INFO] shots train/val/test = {len(train_shots_)}/{len(val_shots_)}/{len(test_shots_)}")
