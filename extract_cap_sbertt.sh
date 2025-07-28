CUDA_VISIBLE_DEVICES=0 python src/extract_cap_sbert.py \
    --caption-folder='../VAD_dataset/UCF-Crimes/Extracted_Captions_janus' \
    --caption-feat-folder='../VAD_dataset/UCF-Crimes/t5_Caption_Features_janus' &

CUDA_VISIBLE_DEVICES=1 python src/extract_cap_sbert.py \
    --caption-folder='../VAD_dataset/XD-Violence/Extracted_Captions_janus' \
    --caption-feat-folder='../VAD_dataset/XD-Violence/t5_Caption_Features_janus' &