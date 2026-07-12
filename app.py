import gradio as gr
import os
import uuid
import subprocess
import torch

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)

# ---------- weights ----------
HF_BASE = "https://huggingface.co/ThunderVVV/HaWoR/resolve/main"
WEIGHTS = [
    ("external/metric_depth_vit_large_800k.pth", "./thirdparty/Metric3D/weights/"),
    ("external/droid.pth",                       "./weights/external/"),
    ("external/detector.pt",                     "./weights/external/"),
    ("hawor/checkpoints/hawor.ckpt",             "./weights/hawor/checkpoints/"),
    ("hawor/checkpoints/infiller.pt",            "./weights/hawor/checkpoints/"),
    ("hawor/model_config.yaml",                  "./weights/hawor/"),
]

print("Downloading model weights")
for src, dst in WEIGHTS:
    os.makedirs(dst, exist_ok=True)
    target = os.path.join(dst, os.path.basename(src))
    if not os.path.exists(target):
        subprocess.run(["wget", "-q", f"{HF_BASE}/{src}", "-P", dst], check=True)

# ---------- MANO (private repo, license-restricted) ----------
from huggingface_hub import hf_hub_download

MANO_REPO = os.environ.get("MANO_REPO", "NathanPereira/mano-private")
HF_TOKEN = os.environ.get("HF_TOKEN")

os.makedirs("./_DATA/data/mano", exist_ok=True)
os.makedirs("./_DATA/data_left/mano_left", exist_ok=True)

if HF_TOKEN:
    hf_hub_download(MANO_REPO, "MANO_RIGHT.pkl", token=HF_TOKEN,
                    local_dir="./_DATA/data/mano")
    hf_hub_download(MANO_REPO, "MANO_LEFT.pkl", token=HF_TOKEN,
                    local_dir="./_DATA/data_left/mano_left")
else:
    print("WARNING: HF_TOKEN not set — MANO weights missing, run_mano() will fail.")

# ---------- imports that depend on the above ----------
import numpy as np
import joblib
import cv2
import imageio
from easydict import EasyDict
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import lookat_matrix, run_vis2_on_video, run_vis2_on_video_cam
from lib.vis.renderer_world import Renderer


def render_reconstruction(input_video, img_focal):
    args = EasyDict()
    args.video_path = input_video
    args.input_type = 'file'
    args.checkpoint = './weights/hawor/checkpoints/hawor.ckpt'
    args.infiller_weight = './weights/hawor/checkpoints/infiller.pt'
    args.vis_mode = 'world'
    args.img_focal = img_focal

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    chunk_path = f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'
    if os.path.exists(chunk_path):
        print("skip hawor motion estimation")
        frame_chunks_all = joblib.load(chunk_path)
        img_focal = args.img_focal
    else:
        frame_chunks_all, img_focal = hawor_motion_estimation(
            args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(
        seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)

    return infiller_and_vis(args, start_idx, end_idx, frame_chunks_all,
                            R_w2c, t_w2c, R_c2w, t_c2w, seq_folder, imgfiles)


def infiller_and_vis(args, start_idx, end_idx, frame_chunks_all,
                     R_w2c_sla_all, t_w2c_sla_all,
                     R_c2w_sla_all, t_c2w_sla_all, seq_folder, imgfiles):
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all)

    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

    faces = get_mano_faces()
    faces_new = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    hand_idx = hand2idx['right']
    pred_glob_r = run_mano(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_dict = {'vertices': pred_glob_r['vertices'][0].unsqueeze(0),
                  'faces': faces_right}

    faces_left = faces_right[:, [0, 2, 1]]
    hand_idx = hand2idx['left']
    pred_glob_l = run_mano_left(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_dict = {'vertices': pred_glob_l['vertices'][0].unsqueeze(0),
                 'faces': faces_left}

    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())

    img = cv2.imread(imgfiles[0])
    renderer = Renderer(img.shape[1], img.shape[0], 1800, 'cuda',
                        bin_size=128, max_faces_per_bin=20000)

    output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
    os.makedirs(output_pth, exist_ok=True)
    image_names = imgfiles[vis_start:vis_end]
    print(f"vis {vis_start} to {vis_end}")

    faces_left_t = torch.from_numpy(faces_left).cuda()
    faces_right_t = torch.from_numpy(faces_right).cuda()
    faces_all = torch.stack((faces_left_t, faces_right_t))

    side_source = torch.tensor([0.463, -0.478, 2.456])
    side_target = torch.tensor([0.026, -0.481, -3.184])
    up = torch.tensor([1.0, 0.0, 0.0])
    view_camera = lookat_matrix(side_source, side_target, up)
    cam_R = view_camera[:3, :3].unsqueeze(0).cuda()
    cam_T = view_camera[:3, 3].unsqueeze(0).cuda()

    out_path = f'{seq_folder}/vis_output_{uuid.uuid4()}.mp4'
    writer = imageio.get_writer(out_path, fps=30, mode='I',
                                format='FFMPEG', macro_block_size=1)
    renderer.set_ground(100, 0, 0)

    for img_i, _ in enumerate(image_names):
        vertices_left = left_dict['vertices'][:, img_i]
        vertices_right = right_dict['vertices'][:, img_i]
        cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
        verts_color = torch.tensor([0.207, 0.596, 0.792, 1.0]).unsqueeze(0).repeat(2, 1)
        vertices_i = torch.stack((vertices_left, vertices_right))
        rend, _ = renderer.render_multiple(
            vertices_i.cuda(), faces_all.cuda(), verts_color.cuda(), cameras, lights)
        writer.append_data(rend)

    writer.close()
    print("finish")
    return out_path


header = ('''
<div class="embed_hidden" style="text-align: center;">
    <h1><b>HaWoR</b>: World-Space Hand Motion Reconstruction from Egocentric Videos</h1>
    <h3>
        Jinglei Zhang<sup>1</sup>,
        <a href="https://jiankangdeng.github.io/" target="_blank">Jiankang Deng</a><sup>2</sup>,
        <a href="https://scholar.google.com/citations?user=syoPhv8AAAAJ&hl=en" target="_blank">Chao Ma</a><sup>1</sup>,
        <a href="https://rolpotamias.github.io" target="_blank">Rolandos Alexandros Potamias</a><sup>2</sup>
    </h3>
    <h3><sup>1</sup>Shanghai Jiao Tong University; <sup>2</sup>Imperial College London</h3>
</div>
<div style="display:flex; gap:0.3rem; justify-content:center;" align="center">
<a href='https://arxiv.org/abs/2501.02973'><img src='https://img.shields.io/badge/Arxiv-2501.02973-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://hawor-project.github.io/'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a>
<a href='https://github.com/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</div>
''')

with gr.Blocks(title="HaWoR", css=".gradio-container") as demo:
    gr.Markdown(header)
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input video", sources=["upload"])
            img_focal = gr.Number(label="Focal Length", value=600)
            submit = gr.Button("Submit", variant="primary")
        with gr.Column():
            reconstruction = gr.Video(label="Reconstruction", show_download_button=True)

        submit.click(fn=render_reconstruction,
                     inputs=[input_video, img_focal],
                     outputs=[reconstruction])

    gr.Examples([
        ['./example/video_0.mp4'],
        ['./example/segment_037.mp4'],
        ['./example/segment_018.mp4'],
    ], inputs=input_video)

demo.launch(server_name="0.0.0.0", server_port=7860)