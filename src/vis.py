import os
import textwrap

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np

def find_closest_key_value(d, frame_idx):
    sorted_items = sorted(
        (int(key), dict(value)) for key, value in d.items() if int(key) <= frame_idx
    )
    return sorted_items[-1] if sorted_items else (None, None)

import os

def npy_path_to_frame_path(npy_path, frames_base_dir):
    class_folder = os.path.basename(os.path.dirname(npy_path))
    base_filename = os.path.basename(npy_path)
    video_name = base_filename.split('__')[0]
    frame_path = os.path.join(frames_base_dir, class_folder, video_name)

    return frame_path

def visualize_video(
    video_name,
    annotation_intervals,
    video_scores,
    video_path,
    video_fps,
    save_path,
    normal_label,
    imagefile_template,
    font_size=18,
):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    video_writer = None

    x = np.arange(len(video_scores))
    ax3.plot(x, video_scores, color="#4e79a7", linewidth=1)
    ymin, ymax = 0, 1
    xmin, xmax = 0, len(video_scores)
    ax3.set_xlim([xmin, xmax])
    ax3.set_ylim([ymin, ymax])
    title = video_name

    # 여기서 annotation_intervals가 주어지면, 그 구간에 빨간 사각형을 그림
    if annotation_intervals is not None:
        for (start, end) in annotation_intervals:
            # 사각형을 그릴 구간: 시작 인덱스 start, 길이 (end-start)
            rect = plt.Rectangle((start, ymin), end - start, ymax - ymin, color="#e15759", alpha=0.5)
            ax3.add_patch(rect)

    ax3.text(0.02, 0.90, title, fontsize=16, transform=ax3.transAxes)
    for y_value in [0.25, 0.5, 0.75]:
        ax3.axhline(y=y_value, color="grey", linestyle="--", linewidth=0.8)

    ax3.set_yticks([0.25, 0.5, 0.75])
    ax3.tick_params(axis="y", labelsize=16)
    ax3.set_ylabel("Anomaly score", fontsize=font_size)
    ax3.set_xlabel("Frame number", fontsize=font_size)
    previous_line = None

    for i, score in enumerate(video_scores):
        ax1.set_title("Video frame", fontsize=font_size)

        img_name = imagefile_template.format(i)
        img_path = os.path.join(video_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.axis("off")

        ax2.text(
            0.5,
            0.5,
            video_name,
            fontsize=18,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                boxstyle="round",
                pad=0.5,
                edgecolor="black",
                linewidth=2,
            ),
            transform=ax2.transAxes,
            wrap=True,
        )
        ax2.axis("off")

        # Update or create the axvline
        if previous_line is not None:
            # Clear previous axvline
            previous_line.remove()

        axvline = ax3.axvline(x=i, color="red")

        fig.tight_layout()

        if video_writer is None:
            fig_size = fig.get_size_inches() * fig.dpi
            video_width, video_height = int(fig_size[0]), int(fig_size[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(save_path), fourcc, video_fps.item(), (video_width, video_height)
            )

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)

        ax1.cla()
        ax2.cla()

        # Update previous_line
        previous_line = axvline

    plt.close()
    video_writer.release()
    cv2.destroyAllWindows()