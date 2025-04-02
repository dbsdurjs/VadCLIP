<<<<<<< HEAD
# idea6(all_caption) branch
# caption 사용 : clip feat = np.concatenate(clip feat, cap feat)
# 부족한 부분의 padding을 마지막 부분의 값들로 pad

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf.pth' --checkpoint-path='../vadclip_pth/model/checkpoint.pth'
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf.pth' --checkpoint-path '../vadclip_pth/model/checkpoint.pth'

=======
>>>>>>> f6fc2de74e0a005ead4e699014ddfeb485b3590f
python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(idea66-1).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(idea66-1).pth' --using-caption
python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(idea66-1).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(idea66-1).pth' --using-caption --save-test-result

python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(idea66-1).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(idea66-1).pth' --using-caption
python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(idea66-1).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(idea66-1).pth' --using-caption --save-test-result

