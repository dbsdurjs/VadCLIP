# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=1 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=1 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=2 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=2 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=3 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=3 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=4 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=4 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=5 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=5 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=6 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=6 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=7 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=7 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=8 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --visual-head=8 --using-caption --save-test-result


# server 컴 실험

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp19).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp19).pth' --using-caption --save-test-result
