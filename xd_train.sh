CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(server_exp10).pth' --checkpoint-path='../vadclip_pth/model/xd_checkpoint_caption(server_exp10).pth' --using-caption
CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(server_exp10).pth' --checkpoint-path '../vadclip_pth/model/xd_checkpoint_caption(server_exp10).pth' --using-caption --save-test-result

# idea66(29) crossvit적용