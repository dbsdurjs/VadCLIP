CUDA_VISIBLE_DEVICES=0 python src/extract_cap_feat.py \
    --caption-folder='../VAD_dataset/UCF-Crimes/Extracted_Captions_janus' \
    --caption-feat-folder='../VAD_dataset/UCF-Crimes/sbert_Caption_Features_janus' &

CUDA_VISIBLE_DEVICES=1 python src/extract_cap_feat.py \
    --caption-folder='../VAD_dataset/XD-Violence/Extracted_Captions_janus' \
    --caption-feat-folder='../VAD_dataset/XD-Violence/sbert_Caption_Features_janus' &