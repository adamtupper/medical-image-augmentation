import argparse
import itertools
import os

import monai.transforms as monai_transforms
import numpy as np
import optuna
import pytorch_lightning as pl
import scipy.stats as stats
import torchvision.transforms.v2 as v2
import yaml
from optuna.storages import JournalFileStorage, JournalStorage
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from model import LitImageClassifier

AUGMENTATIONS = {
    "random_crop": v2.RandomCrop(224),
    "horizontal_flip": v2.RandomHorizontalFlip(),
    "vertical_flip": v2.RandomVerticalFlip(),
    "rotation": v2.RandomRotation(degrees=30),
    "translate_x": v2.RandomAffine(degrees=0, translate=[0.2, 0]),
    "translate_y": v2.RandomAffine(degrees=0, translate=[0, 0.2]),
    "shear_x": v2.RandomAffine(degrees=0, shear=[0.0, 30.0]),
    "shear_y": v2.RandomAffine(degrees=0, shear=[0.0, 0.0, 0.0, 30.0]),
    "brightness": v2.ColorJitter(brightness=0.5),
    "contrast": v2.ColorJitter(contrast=0.5),
    "saturation": v2.ColorJitter(saturation=0.5),
    "gaussian_blur": v2.GaussianBlur(kernel_size=3),
    "equalize": v2.RandomEqualize(),
    "median_blur": v2.RandomApply([monai_transforms.MedianSmooth(radius=3)], p=0.5),
    "grid_distortion": monai_transforms.RandGridDistortion(prob=1.0),
    "gaussian_noise": monai_transforms.RandGaussianNoise(prob=0.5),
    "scaling": v2.RandomAffine(
        degrees=0, scale=[0.8, 1.2]
    ),  # Sensible range based on prior works
    "elastic_transform": v2.ElasticTransform(),
}


def get_class_balanced_weights(samples_per_class, beta=0.9999):
    """Calculate the class weights using the Class-Balanced Loss proposed by Cui et al.
    (2019): https://arxiv.org/abs/1901.05555.

    This version is modified so that the number of samples per class is >= 1. This
    accounts for the fact that some classes may not be present in the labeled training
    set.
    """
    samples_per_class = np.maximum(samples_per_class, 1)
    effective_num = 1.0 - np.power(beta, samples_per_class)

    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_class)

    return weights


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="individual_effects.py")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path to the dataset directory",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        help="The directory containing the configuration files",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="The directory to save logs and checkpoints to",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a single trial in `fast_dev_run` mode to quickly verify everything is working",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="The number of workers to use for data loading",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The random seed to use for KFold CV"
    )
    parser.add_argument(
        "--dataset", type=str, default="busi", help="The dataset to use"
    )

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.dataset)

    # Load dataset-specific configuration
    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml"), "r") as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    return args


def get_dataset_folds(args, train_transform, val_transform):
    if args.dataset in ["bus_bra", "bus_bra_birads"]:
        # The dataset is already divided into 5 different stratified train/val splits
        for split in range(5):
            for train_fold, val_fold in [[0, 1], [1, 0]]:
                train_data = ImageFolder(
                    os.path.join(args.data_dir, f"split{split}", f"fold{train_fold}"),
                    transform=train_transform,
                )
                val_data = ImageFolder(
                    os.path.join(args.data_dir, f"split{split}", f"fold{val_fold}"),
                    transform=val_transform,
                )
                yield split, train_fold, train_data, val_data
    else:
        # Repeat Stratified K-fold CV 5 times
        train_dataset = ImageFolder(
            os.path.join(args.data_dir), transform=train_transform
        )
        val_dataset = ImageFolder(os.path.join(args.data_dir), transform=val_transform)

        for split in range(5):
            k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=split)
            for train_fold, (train_idxs, val_idxs) in enumerate(
                k_fold.split(X=np.zeros(len(train_dataset)), y=train_dataset.targets)
            ):
                train_data = Subset(train_dataset, train_idxs)
                val_data = Subset(val_dataset, val_idxs)
                yield split, train_fold, train_data, val_data


def objective(trial, args):
    """Evaluate a set of hyperparameter values using cross-validation."""
    augmentation_pairs = [
        ",".join(x) for x in itertools.permutations(AUGMENTATIONS.keys(), 2)
    ]
    augmentation_pairs = [x for x in augmentation_pairs if "elastic_transform" in x]
    pair = trial.suggest_categorical("transforms", augmentation_pairs).split(",")
    operation1, operation2 = AUGMENTATIONS[pair[0]], AUGMENTATIONS[pair[1]]

    # Define the train and val transforms
    if "random_crop" in pair:
        train_transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(224, antialias=True),
                operation1,
                operation2,
                v2.Normalize(args.mean, args.std),
            ]
        )
    else:
        train_transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(224, antialias=True),
                v2.CenterCrop(224),
                operation1,
                operation2,
                v2.Normalize(args.mean, args.std),
            ]
        )

    val_transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(224, antialias=True),
            v2.CenterCrop(224),
            v2.Normalize(args.mean, args.std),
        ]
    )

    # 5 x 2 cross validation
    balanced_accuracies = []
    for split, train_fold, train_subset, val_subset in get_dataset_folds(
        args, train_transform, val_transform
    ):
        pl.seed_everything(args.seed)

        # Define data loaders for training and validation data
        train_loader = DataLoader(
            train_subset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers,
        )

        # Calculate the class weights
        counts = np.bincount(
            (
                train_subset.dataset.targets
                if type(train_subset) == Subset
                else train_subset.targets
            ),
            minlength=args.num_classes,
        )
        class_weights = get_class_balanced_weights(counts)

        # Instantiate the model
        lit_model = LitImageClassifier(
            resnet18, num_classes=args.num_classes, class_weights=class_weights
        )

        # Setup the trainer
        checkpoint_callback = ModelCheckpoint(
            monitor="val/bacc_epoch", mode="max", save_top_k=1
        )
        trainer = pl.Trainer(
            max_epochs=100,
            deterministic=False,
            default_root_dir=os.path.join(
                args.log_dir,
                f"{pair[0]}-{pair[1]}",
                f"split_{split}",
                f"trainfold_{train_fold}",
            ),
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback],
        )

        # Train the model
        trainer.fit(lit_model, train_loader, val_loader)
        balanced_accuracies.append(lit_model.val_bacc_best.item())

    # Return a simple mean and std. err. of the balanced accuracies just for Optuna
    return np.mean(balanced_accuracies), stats.sem(balanced_accuracies)


def main():
    args = parse_args()

    # Configure Optuna storage
    os.makedirs(args.log_dir, exist_ok=True)
    storage_path = os.path.join(args.log_dir, f"{args.dataset}.log")
    storage = JournalStorage(JournalFileStorage(storage_path))

    # Configure study
    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
        study_name=args.dataset,
        storage=storage,
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )

    # Run the study
    study.optimize(
        lambda trial: objective(trial, args), n_trials=1
    )  # Use len(list(itertools.permutations(AUGMENTATIONS.keys(), 2))) for full study

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    for trial in study.best_trials:
        print("  Value: {}".format(trial.values))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
