CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp11).pth' --checkpoint-path='../vadclip_pth/model/ucf_checkpoint_caption(server_exp11).pth' --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp11).pth' --checkpoint-path '../vadclip_pth/model/ucf_checkpoint_caption(server_exp11).pth' --using-caption --save-test-result

# idea66(29) crossvit적용