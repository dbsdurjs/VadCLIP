# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp6).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(server_exp6).pth' --using-caption
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp6).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(server_exp6).pth' --using-caption --save-test-result

python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(server_exp6).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(server_exp6).pth' --using-caption
python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(server_exp6).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(server_exp6).pth' --using-caption --save-test-result

