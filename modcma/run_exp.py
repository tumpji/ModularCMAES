from modcma import evaluate_bbob


def main(
        *,
        fid: int,
        dim: int,
        data_folder: str,
        exp_repetitions: int = 1,
        **parameters
):
    evaluate_bbob(
        fid,
        dim,
        logging=True,
        data_folder=data_folder,
        iterations=exp_repetitions,
        **parameters
    )


params = {
    'surrogate_model': 'Linear'
}

main(dim=2, fid=1, data_folder=".", **params)
