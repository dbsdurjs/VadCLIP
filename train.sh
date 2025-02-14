# idea4(all_caption) branch
# caption 사용 : clip feat = clip feat + alpha*cap feat
# alpha 값을 조절하면서 clip feat와 더하는 방식

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf.pth' --checkpoint-path='../vadclip_pth/model/checkpoint.pth'
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf.pth' --checkpoint-path '../vadclip_pth/model/checkpoint.pth'

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(alpha=1).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(alpha=1).pth' --using-caption
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(alpha=1).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(alpha=1).pth' --using-caption

# python src/ucf_test.py --model-path '../vadclip_pth/model_ucf_pretrained.pth'


