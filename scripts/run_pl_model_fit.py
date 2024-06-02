import argparse

from massspecgym.runner import init_run


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="path to template config file"
    )
    parser.add_argument(
        "-c",
        "--custom_fp",
        type=str,
        required=False
    )
    parser.add_argument(
        "-w",
        "--wandb_mode",
        type=str, 
        default="online",
        choices=["online","offline","disabled"]
    )
    args = parser.parse_args()

    init_run(args.template_fp, args.custom_fp, args.wandb_mode)