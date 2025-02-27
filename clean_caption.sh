# Set the UCF Crime directory
ucf_crime_dir="/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes"

# Set paths
root_path="${ucf_crime_dir}/Extracted_Frames"
annotationfile_path="/home/yeogeon/YG_main/diffusion_model/VadCLIP/list/Anomaly_Train.txt"
batch_size=64
frame_interval=16
fps=30
clip_duration=10
num_samples=10
num_neighbors=1

# # Activate the virtual environment
# VENV_DIR="/path/to/venv/lavad"
# # shellcheck source=/dev/null
# source "$VENV_DIR/bin/activate"

captions_dir_template="$ucf_crime_dir/Extracted_Frames_captions/"
index_dir="$ucf_crime_dir/create_index/"
output_dir="${ucf_crime_dir}/captions/clean/"
python src/image_text_caption_cleaner.py \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "$output_dir" \
    --captions_dir_template "${captions_dir_template}" \
    --index_dir "${index_dir}" \
    --fps "$fps" \
    --clip_duration "$clip_duration" \
    --num_samples "$num_samples" \
    --num_neighbors "$num_neighbors"