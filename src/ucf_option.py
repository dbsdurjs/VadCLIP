import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int) # 1(87.23%, 5.67%) or 2(87.08%, 6.22%)
parser.add_argument('--visual-layers', default=2, type=int) # 2(87.23%, 5.67%) or 4(87%, 7.31%) (6부터는 메모리 초과)
parser.add_argument('--attn-window', default=16, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--text-layers', default=8, type=int) # 8일때 87.23%
parser.add_argument('--text-dim', default=512, type=int)
parser.add_argument('--text-head', default=1, type=int) # 1일때 87.23%
parser.add_argument('--lstm-layer', default=1, type=int) # 1일때 87.23%
parser.add_argument('--cross-attn-head', default=1, type=int) # 1 - 87.23%, 5.67% | 2 - 87.11%, 6.27%

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='../vadclip_pth/model/model_ucf_caption(server_exp27).pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='../vadclip_pth/model/checkpoint_ucf_caption.pth')
parser.add_argument('--batch-size', default=64, type=int)

parser.add_argument('--using-caption', action='store_true', default=True)
parser.add_argument('--saved-video', action='store_true', default=False)
parser.add_argument('--save-test-result', action='store_true', default=False)

parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')

parser.add_argument('--train-cap-list', default='list/ucf_CLIP_rgb_description_refine_2.csv')
parser.add_argument('--test-cap-list', default='list/ucf_CLIP_rgbtest_description.csv')

parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')
parser.add_argument('--gt-txt', default='./list/Temporal_Anomaly_Annotation.txt')
parser.add_argument('--frame_base_folder', default='../VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])