import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int) # 4일때 제일 나은듯?
parser.add_argument('--visual-layers', default=1, type=int) # 1일때 제일 나은듯? (6부터는 메모리 초과)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--text-layers', default=1, type=int) # xd에서는 1일때가 가장 좋음(or 2)
parser.add_argument('--text-dim', default=512, type=int)
parser.add_argument('--text-head', default=1, type=int) # 1일때가 가장 좋음(1,2,4,8 중 별 차이 없긴함), head는 text dim으로 나누어떨어져야 함
parser.add_argument('--lstm-layer', default=5, type=int) # 5일때 가장 좋음(81%, 25%)
parser.add_argument('--cross-attn-head', default=1, type=int) 

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='../vadclip_pth/model/model_xd_caption.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='../vadclip_pth/model/checkpoint_xd_caption.pth')
parser.add_argument('--batch-size', default=96, type=int)

parser.add_argument('--using-caption', action='store_true', default=True)
parser.add_argument('--saved-video', action='store_true', default=False)
parser.add_argument('--save-test-result', action='store_true', default=False)

parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')

parser.add_argument('--train-cap-list', default='list/xd_CLIP_rgb_description_refine_2.csv')
parser.add_argument('--test-cap-list', default='list/xd_CLIP_rgbtest_description.csv')

parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])