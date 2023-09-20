import argparse
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from alignment_utils import (
    gaussian_projection,
    get_accuracies,
    get_loss_func,
    get_one_to_one_features,
    get_visual_baselines,
    l2_norm,
    load_features,
    sinkhorn_assignment,
    tensors_to_device,
)
from save_features import N_AUGMENTATIONS, get_config
from utils import feature_augmentation, seed_everything


def procrustes_align(
        features_src: torch.Tensor,
        features_tgt: torch.Tensor,
        beta: float = 0.85) -> torch.Tensor:

    u, _, v = torch.svd(features_src.T @ features_tgt)
    W = u @ v.T

    identity = torch.eye(W.size(0)).to(W.device)
    W = W - (W - identity) * beta
    return W


def pseudo_align(features_src: torch.Tensor,
                 features_tgt: torch.Tensor) -> torch.Tensor:

    # solve the least squre with pseudo inverse
    # no orthgonality
    x_source_pseudo = torch.linalg.inv(features_src.T @ features_src)
    x_source_pseudo = x_source_pseudo @ features_src.T
    W = x_source_pseudo @ features_tgt
    return W


def log_rampup(current: float, rampup_length: int) -> float:
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1 - np.exp(-5.0 * current / rampup_length))


def spectral_projection(trasfm: torch.Tensor) -> torch.Tensor:
    u, s, v = torch.svd(trasfm)
    s[s > 1] = 1.0
    s[s < 0] = 0.0
    trasfm = u @ (torch.diag(s) @ v.T)
    return trasfm


def cross_validation_beta_procrustes(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    num_of_tries: int = 25,
    num_samples: int = 3,
    five_crop: bool = False,
) -> torch.Tensor:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    visual_features, class_prototypes, labels = tensors_to_device(
        [visual_features, class_prototypes, labels], device
    )

    betas: torch.Tensor = torch.linspace(0.0, 1.0, num_of_tries)
    best_beta = 0.0
    beta_transform = None
    max_score = float("-inf")

    if five_crop:
        # only keep center crops - faster more stable
        mask = (
            torch.tensor([0, 0, 0, 0, 1]).repeat(
                visual_features.shape[0] // N_AUGMENTATIONS).bool()
        ).to(device)
        visual_features = visual_features[mask]
        labels = labels[mask]

    def create_arrrays(
        visual_features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:

        new_train_size = int(len(visual_features) * 0.8)

        new_train_array = {}
        new_test_array = {}

        new_train_array["visual_features"] = visual_features[:new_train_size]
        new_train_array["labels"] = labels[:new_train_size]
        new_train_array["text_features"] = class_prototypes

        new_test_array["visual_features"] = visual_features[new_train_size:]
        new_test_array["labels"] = labels[new_train_size:]
        new_test_array["text_features"] = class_prototypes
        new_test_array = [new_test_array]

        return new_train_array, new_test_array

    for beta in tqdm(betas):
        score = 0.0
        for _ in range(num_samples):
            perm = torch.randperm(len(visual_features))
            new_train_array, new_test_array = create_arrrays(
                visual_features[perm], labels[perm]
            )

            transfm = procrustes_align(
                new_train_array["visual_features"],
                new_train_array["text_features"][new_train_array["labels"]],
                beta=beta.item(),
            )

            acc = get_accuracies(
                new_train_array, new_test_array, transform=transfm)

            score += acc["test"]["top_1"]
        score = score / num_samples

        if score > max_score:
            max_score = score
            best_beta = beta

    logger.info(f"Beta selected is: {best_beta:.2f}")

    beta_transform = procrustes_align(
        visual_features, class_prototypes[labels], beta=best_beta
    )

    return beta_transform.cpu()


def iterative_unsupervised_refinement(
    args: argparse.Namespace,
    loss_func: Callable,
    train_arrays: Dict[str, torch.Tensor],
    test_arrays: List[Dict[str, torch.Tensor]],
    th: float = 0.0,
) -> torch.Tensor:

    n_unsup_iters = args.n_unsup_iters
    class_prototypes = train_arrays["text_features"]
    N, _ = train_arrays["visual_features"].size()

    soft_assignments = sinkhorn_assignment(
        train_arrays["visual_features"],
        class_prototypes,
        blur=0.05,
    )

    mask = soft_assignments.max(-1)[0] > th
    soft_assignments = soft_assignments[mask]
    visual_features = train_arrays["visual_features"][mask]
    labels = soft_assignments.argmax(-1)

    transfm = cross_validation_beta_procrustes(
        visual_features,
        class_prototypes,
        labels,
        five_crop=args.five_crop,
    )

    accuracies = get_accuracies(
        train_arrays, test_arrays, transform=transfm, five_crop=args.five_crop
    )

    logger.info(f"Initial results: {accuracies}\n")

    tbar = tqdm(range(n_unsup_iters))

    for n_iter in tbar:
        soft_assignments = sinkhorn_assignment(
            l2_norm(train_arrays["visual_features"] @ transfm),
            class_prototypes,
            blur=0.05,
        )

        mask = soft_assignments.max(-1)[0] > th

        soft_assignments = soft_assignments[mask]
        visual_features = train_arrays["visual_features"][mask]
        text_features = l2_norm(soft_assignments @ class_prototypes)
        labels = soft_assignments.argmax(-1)

        transfm = mapping_refinement(
            args,
            loss_func=loss_func,
            init_transfm=transfm,
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            train_visual_feats=visual_features,
            train_text_feats=text_features,
            class_prototypes=class_prototypes,
            labels=labels,
            batch_size=len(visual_features),
            verbose=True,
        )

        accuracies = get_accuracies(
            train_arrays,
            test_arrays,
            transform=transfm,
            five_crop=args.five_crop,
        )

        tbar.set_description(
            f"Iter: {n_iter} | Used examples: {mask.sum()}/{N} - Acc {accuracies}"
        )

    logger.info(f"After refinement results: {accuracies}\n")

    return transfm


def mapping_refinement(
    args: argparse.Namespace,
    loss_func: Callable,
    init_transfm: torch.Tensor,
    train_arrays: Dict[str, torch.Tensor],
    test_arrays: List[Dict[str, torch.Tensor]],
    train_visual_feats: torch.Tensor,
    train_text_feats: Optional[torch.Tensor] = None,
    class_prototypes: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    momentum: float = 0.9,
    return_ema_transform: bool = False,
    verbose: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema_transform = torch.eye(init_transfm.size(1)).to(device)

    num_instances = (
        train_visual_feats.size(0) // N_AUGMENTATIONS
        if args.five_crop
        else train_visual_feats.size(0)
    )

    if batch_size is None:
        batch_size = num_instances if args.five_crop else int(
            num_instances * 0.75)

    assert (class_prototypes is not None and labels is not None) or train_text_feats

    transfm = torch.nn.Parameter(
        init_transfm.clone().to(device), requires_grad=True)
    class_prototypes = (
        class_prototypes.to(device) if class_prototypes is not None else None
    )

    opt = torch.optim.AdamW(
        [transfm],
        lr=args.learning_rate,
        eps=1e-08,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # init learning rate scheduler, cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.n_iters, eta_min=args.cosine_end_lr
    )

    tbar = tqdm(range(args.n_iters)) if verbose else range(args.n_iters)
    for num_iter in tbar:
        opt.zero_grad()

        if args.five_crop:
            # use only one of the crops at random (faster)
            mask = torch.tensor([0, 0, 0, 0, 1]).bool()
            mask = mask[torch.randperm(5)].repeat(num_instances)

            train_visual_batch = train_visual_feats[mask]
            labels_batch = labels[mask] if labels is not None else None
            train_text_batch = (
                train_text_feats[mask] if train_text_feats is not None else None
            )
        else:
            train_visual_batch = train_visual_feats
            train_text_batch = train_text_feats
            labels_batch = labels

        if batch_size is not None:
            batch_indices = torch.randperm(num_instances)[:batch_size]

            train_visual_batch = train_visual_batch[batch_indices]
            train_text_batch = (
                train_text_batch[batch_indices]
                if train_text_batch is not None
                else None
            )
            labels_batch = labels_batch[batch_indices] if labels is not None else None

        train_visual_batch, train_text_batch, labels_batch = tensors_to_device(
            [train_visual_batch, train_text_batch, labels_batch], device
        )

        if args.interpolate_features:
            train_visual_batch, labels_batch = feature_augmentation(
                train_visual_batch, labels_batch
            )

        if args.gaussian_noise > 0.0 and np.random.uniform() > 0.5:
            train_visual_batch += (
                torch.randn_like(train_visual_batch) * args.gaussian_noise
            )

        if args.dropout > 0.0 and np.random.uniform() > 0.5:
            train_visual_batch = torch.nn.functional.dropout(
                train_visual_batch, p=args.dropout
            )

        # compute the loss
        loss = loss_func(
            visual_features=train_visual_batch @ transfm,
            text_features=train_text_batch,
            class_prototypes=class_prototypes,
            labels=labels_batch,
            knn=args.knn,
        )

        # update the parameters
        loss.backward()
        opt.step()
        scheduler.step()

        # project the transformation matrix to the space of orthogonal matrices
        if args.spectral_proj:
            transfm.data = spectral_projection(transfm.data)
        elif args.orthogonalize:
            transfm.data = (1 + args.orth_beta) * transfm.data - args.orth_beta * (
                (transfm.data @ transfm.data.T) @ transfm.data
            )

        momentum = 0.1 * log_rampup(num_iter, args.n_iters // 2) + 0.9
        ema_transform = momentum * ema_transform.data + \
            (1.0 - momentum) * transfm.data

        # Compute train and test accuracies
        accuracies = get_accuracies(
            train_arrays, test_arrays, transform=transfm.cpu(), five_crop=args.five_crop
        )

        # update the progress bar
        if verbose:
            tbar.set_description(f"Loss: {loss.item():.4f} | {accuracies}")

    if return_ema_transform:
        return transfm.detach().cpu(), ema_transform.cpu()

    return transfm.detach().cpu()


def main(args, cfg):
    # ==================================================================
    logger.info("Loading features...")
    train_arrays, test_arrays = load_features(cfg, args)

    if args.model_type in ["mocov3", "barlowtwins", "byol"]:
        baselines = get_visual_baselines(train_arrays, test_arrays)
        logger.info(f"Baseline results are {baselines}")

        train_arrays, test_arrays = gaussian_projection(
            train_arrays, test_arrays)

    # ==================================================================

    logger.info("Baseline clip resutls are:")

    accuracies = get_accuracies(
        train_arrays, test_arrays, transform=None, five_crop=args.five_crop
    )
    logger.info(f"Clip results {accuracies}\n")

    # ==================================================================

    if not args.unsupervised:

        train_text_feats = get_one_to_one_features(
            train_arrays["visual_features"],
            train_arrays["text_features"],
            train_arrays["labels"],
        )

        logger.info("Procrustes results:")

        transfm_proc = procrustes_align(
            train_arrays["visual_features"].cuda(),
            train_text_feats.cuda(),
            beta=0.0,
        )

        accuracies = get_accuracies(
            train_arrays,
            test_arrays,
            transform=transfm_proc,
            five_crop=args.five_crop,
        )

        logger.info(f"Procrustes results: {accuracies}\n")

        if args.beta_procrustes is None:
            transfm = cross_validation_beta_procrustes(
                train_arrays["visual_features"],
                train_arrays["text_features"],
                train_arrays["labels"],
                five_crop=args.five_crop,
            )
        else:
            transfm = procrustes_align(
                train_arrays["visual_features"].cuda(),
                train_text_feats.cuda(),
                beta=args.beta_procrustes,
            )

        accuracies = get_accuracies(
            train_arrays, test_arrays, transform=transfm, five_crop=args.five_crop
        )

        logger.info(f"Beta-Procrustes results: {accuracies}\n")

        # ==================================================================

        logger.info("Mapping refinement ...")

        refined_transfm, ema_transform = mapping_refinement(
            args,
            loss_func=get_loss_func(args),
            init_transfm=transfm,
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            train_visual_feats=train_arrays["visual_features"],
            train_text_feats=train_text_feats,
            class_prototypes=train_arrays["text_features"],
            labels=train_arrays["labels"],
            batch_size=args.batch_size,
            return_ema_transform=True,
        )

        accuracies = get_accuracies(
            train_arrays,
            test_arrays,
            transform=refined_transfm,
            five_crop=args.five_crop,
        )
        logger.info(f"After refinement results: {accuracies}\n")

        if cfg.DATA.USE_BASE_AND_NEW or cfg.DATA.TARGET_DATASET is not None:

            accuracies = get_accuracies(
                train_arrays,
                test_arrays,
                transform=refined_transfm,
                target_set_transform=ema_transform,
                five_crop=args.five_crop,
            )

            logger.info(f"Two mappings results: {accuracies}\n")

    # ==================================================================

    if args.unsupervised:
        logger.info("Unsupervised iterative adaptation ...")

        refined_transfm = iterative_unsupervised_refinement(
            args,
            loss_func=get_loss_func(args),
            train_arrays=train_arrays,
            test_arrays=test_arrays,
        )
        accuracies = get_accuracies(
            train_arrays,
            test_arrays,
            transform=refined_transfm,
            five_crop=args.five_crop,
        )

        logger.info(f"Unsupervised adaptation results: {accuracies}\n")

    if args.model_type in ["mocov3", "barlowtwins", "byol"]:
        return accuracies, baselines

    return accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--fewshot-path", type=str, default=None)
    parser.add_argument("--model-chekpoint", type=str, default=None)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["clip", "clip_prompt", "mocov3", "barlowtwins",
                 "byol", "align", "flava", "alt_clip"],
        required=True,
    )
    parser.add_argument(
        "--refinement-loss",
        type=str,
        choices=["csls", "adaptive", "contrastive", "triplet"],
        default="adaptive",
    )

    parser.add_argument("--unsupervised", action="store_true")
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--cosine-end-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--n-unsup-iters", type=int, default=5)

    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--arerank-scale", type=float, default=4.0)
    parser.add_argument("--spectral-proj", action="store_true")
    parser.add_argument("--orthogonalize", action="store_true")
    parser.add_argument("--orth-beta", type=float, default=0.01)
    parser.add_argument("--pseudo-align", action="store_true")
    parser.add_argument("--beta-procrustes", type=float, default=None)

    parser.add_argument("--gaussian-noise", type=float, default=0.035)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--interpolate-features", action="store_true")
    parser.add_argument("--five-crop", action="store_true")

    arguments = parser.parse_args()
    configs = get_config(arguments)

    seed_everything(configs.RNG_SEED)
    main(arguments, configs)
