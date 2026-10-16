export HF_ENDPOINT=https://hf-mirror.com

# bash script/sd14_instance.sh
# bash script/sd14_style.sh

# bash script/sdxl_instance.sh
# bash script/sdxl_style.sh

# bash script/sd3_instance.sh
# bash script/sd3_style.sh

ARCHITECTURE=sdxl NUM_GPUS=2 bash /home/shx/code/ce-new/casu-2/script/paper_all.sh

NUM_GPUS=2 bash /home/shx/code/ce-new/casu-2/script/paper_sample_all.sh


bash script/flux_instance.sh
bash script/flux_style.sh