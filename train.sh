# idea6(all_caption) branch
# caption 사용 : clip feat = np.concatenate(clip feat, cap feat)
# 부족한 부분의 padding을 마지막 부분의 값들로 pad
# caption clean 사용(대표 프레임 당 10개 캡션)

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf.pth' --checkpoint-path='../vadclip_pth/model/checkpoint.pth'
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf.pth' --checkpoint-path '../vadclip_pth/model/checkpoint.pth'

python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(padding, clean_caption, no_represen).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(padding, clean_caption, no_represen).pth' --using-caption
python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(padding, clean_caption, no_represen).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(padding, clean_caption, no_represen).pth' --using-caption
