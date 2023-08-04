import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader

from . import vae, capsnet, training
from .train_results import FitResult
from .datasets import GraphemesDataset

#DATA_DIR = os.path.expanduser("~/.pytorch-datasets")
MODEL_TYPES = dict(vae=vae.VariationalAutoencoder, capsnet=capsnet.CapsNet)

def run_experiment(
    run_name,
    data_dir,
    letter=None,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-5,
    # Model params
    model_type="vae",
    model_config=None,
    save_model=False,
    # You can add extra configuration for your experiments here
    **kw,
):
    """
    Executes a single run of an experiment with a single configuration.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = GraphemesDataset(data_dir, train=True, by_letter=letter, transform=tf)
    ds_test = GraphemesDataset(data_dir, train=False, by_letter=letter, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fit_res = None
    # Data - use DataLoader
    train_loader = DataLoader(ds_train, batch_size=bs_train, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=bs_test, shuffle=False)

    if model_type == "vae":
        model = vae.VariationalAutoencoder(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        trainer = training.VAETrainer(model, model.loss, optimizer, device)
        fit_res = trainer.fit(dl_train=train_loader, dl_test=test_loader, 
                              num_epochs=epochs, checkpoints=checkpoints, max_batches=batches, **kw)
    elif model_type == "capsnet":
        model = capsnet.CapsNet(model_config)
        model = torch.nn.DataParallel(model)
        model = model.module
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        trainer = training.CapsNetTrainer(model, model.loss, optimizer, device)
        fit_res = trainer.fit(dl_train=train_loader, dl_test=test_loader, 
                              num_epochs=epochs, checkpoints=checkpoints, max_batches=batches, **kw)

    if save_model:
        save_model_state(model, os.path.join(out_dir, "models"), run_name)

    save_experiment(run_name, out_dir, cfg, fit_res)

def save_model_state(model, out_dir, run_name):
    output_filename = f"{os.path.join(out_dir, run_name)}.pt"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), output_filename)
    print(f"*** Model file {output_filename} written")

def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    output_filename = f"{os.path.join(out_dir, run_name)}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="Deep Graphemics Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # Data Augmentation


    # # Model

    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )
    # configuration for the model instance
    # TODO: figure out how to pass this in
    sp_exp.add_argument(
        "--model-config",
        "-C",
        type=json.loads,
        default=None,
        help="JSON string with model configuration",
    )

    parsed = p.parse_args()
    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
