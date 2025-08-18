# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=1 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=1 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=2 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=2 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=3 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=3 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=4 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=4 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=5 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=5 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=6 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=6 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=7 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=7 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=8 --using-caption
# CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --visual-head=8 --using-caption --save-test-result


# server 컴 실험

CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --using-caption
CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(hackerton).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(hackerton).pth' --using-caption --save-test-result
