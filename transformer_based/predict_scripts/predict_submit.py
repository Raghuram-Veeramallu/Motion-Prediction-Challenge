# chage this if you have problem
import sys

from prerender.utils.utils import get_config
# sys.path.insert(1, "~/.local/lib/python3.6/site-packages")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from submission_proto import motion_submission_pb2
from model.transfomer import MultiPathPP
from model.data import get_dataloader


def main():
    config = get_config(sys.argv[1])

    test_dataloader = get_dataloader(config["test"]["data_config"])
    model_filepath = config["test"]["model_path"]
    results_filepath = config["test"]["results_path"]

    model = MultiPathPP(config["model"])
    model.load_state_dict(torch.load(model_filepath)["model_state_dict"])

    model.cuda()
    model.eval()

    response = {}
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            if config["train"]["normalize"]:
                data = normalize(data, config)
            dict_to_cuda(data)
            probas, coordinates, covariance_matrices, loss_coeff = model(data, 0)
            # confidences_logits, logits = model(data, 0)
            if config["train"]["normalize_output"]:
                coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
            confidences = torch.softmax(confidences_logits, dim=1)

            logits = logits.cpu().numpy()
            confidences = confidences.cpu().numpy()
            agent_id = agent_id.cpu().numpy()
            center = center.cpu().numpy()
            yaw = yaw.cpu().numpy()
            for p, conf, aid, sid, c, y in zip(
                logits, confidences, agent_id, scenario_id, center, yaw
            ):
                if sid not in response:
                    response[sid] = []

                response[sid].append(
                    {"aid": aid, "conf": conf, "pred": p, "yaw": -y, "center": c}
                )

    motion_challenge_submission = motion_submission_pb2.MotionChallengeSubmission()
    motion_challenge_submission.account_name = 'dummy_name'
    motion_challenge_submission.authors.extend(['test_author'])
    motion_challenge_submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    )
    motion_challenge_submission.unique_method_name = 'grp10'

    selector = np.arange(4, 81, 5)
    for scenario_id, data in tqdm(response.items()):
        scenario_predictions = motion_challenge_submission.scenario_predictions.add()
        scenario_predictions.scenario_id = scenario_id
        prediction_set = scenario_predictions.single_predictions

        for d in data:
            predictions = prediction_set.predictions.add()
            predictions.object_id = int(d["aid"])

            y = d["yaw"]
            rot_matrix = np.array([
                [np.cos(y), -np.sin(y)],
                [np.sin(y), np.cos(y)],
            ])

            for i in np.argsort(-d["conf"]):
                scored_trajectory = predictions.trajectories.add()
                scored_trajectory.confidence = d["conf"][i]

                trajectory = scored_trajectory.trajectory

                p = d["pred"][i][selector] @ rot_matrix + d["center"]

                trajectory.center_x.extend(p[:, 0])
                trajectory.center_y.extend(p[:, 1])

    with open(results_filepath, "w+") as f:
        f.write(motion_challenge_submission.SerializeToString())


if __name__ == "__main__":
    main()
