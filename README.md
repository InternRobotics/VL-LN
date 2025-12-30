<div align="center">

![demo](images/teaser.png "demo")
[![HomePage](https://img.shields.io/badge/HomePage-144B9E?logo=ReactOS&logoColor=white)](https://0309hws.github.io/VL-LN.github.io/)
[![Paper](https://img.shields.io/badge/Paper-B31B1B?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2512.22342)
[![Data](https://img.shields.io/badge/Data-FFA500?logo=readthedocs&logoColor=white)](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench/tree/main/)
[![Model](https://img.shields.io/badge/Model-2CA02C?logo=huggingface&logoColor=white)](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel/tree/main)

</div>

# VL-LN Bench

## 🏠 Introduction

VL-LN is a benchmark that provides a large-scale, automatically generated dataset and a comprehensive evaluation protocol for training and assessing dialog-enabled navigation models.

### Contents
- [Data Collection Pipeline](https://github.com/InternRobotics/VL-LN)
- [Training and Evaluation](https://github.com/InternRobotics/InternNav)
- [Dataset](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench) and [Base Model](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel)

## 📚 Getting Started

### 1. Download Data & Assets
- Scene Datasets

  
    Download the scene dataset of [MP3D](https://niessner.github.io/Matterport/)
- [VL-LN Data](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench)
- [VL-LN Base Model](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel)
  
After unzipping the base model, scene datasets, and trajectory data, put everything under VL-LN-Bench/ in the layout below.
  ```bash
  VL-LN-Bench/
  ├── base_model/ 
  │   └── iion/
  ├── raw_data/ 
  │   └── mp3d/
  │       ├── scene_summary/
  │       ├── train/ 
  │       │   ├── train_ion.json.gz
  │       │   └── train_iion.json.gz
  │       └── val_unseen/ 
  │           ├── val_unseen_ion.json.gz
  │           └── val_unseen_iion.json.gz
  ├── scene_datasets/
  │   └── mp3d/
  │       ├── 17DRP5sb8fy/
  │       ├── 1LXtFkjw3qL/
  │       ...
  └── traj_data/
      ├── mp3d_split1/
      ├── mp3d_split2/
      └── mp3d_split3/
  ```

### 2. Environment Setup
- Get Code
  ```bash
  git clone git@github.com:InternRobotics/VL-LN.git # code for data collection
  git clone git@github.com:InternRobotics/InternNav.git # code for training and evaluation
  ```
- Create Conda Environment
  ```bash
  conda create -n vlln python=3.9 -y
  conda activate vlln
  ```
- Install Dependencies
  ```bash
  conda install habitat-sim=0.2.4 withbullet headless -c conda-forge -c aihabitat
  cd VL-LN
  pip install -r requirements.txt
  cd ../InternNav
  pip install -e .
  ```

### 3. Guidance for Data Collection
- Prerequisites:
  - Get pointnav_weights.pth from [VLFM](https://github.com/bdaiinstitute/vlfm/tree/main/data)
  - Arrange the Directory Structure Like This
    ```bash
    VL-LN
    ├── dialog_generation/
    ├── images/
    ├── VL-LN-Bench/
    │   ├── base_model/ 
    │   ├── raw_data/ 
    │   ├── scene_datasets/
    │   ├── traj_data/
    │   └── pointnav_weights.pth
    ...
    ```
- Collect Trajectories
  ```bash
  # If having slurm
  sbatch generate_frontiers_dialog.sh

  # Or directly run
  python generate_frontiers_dialog.py \
      --task instance \
      --vocabulary hm3d \
      --scene_ids all \
      --shortest_path_threshold 0.1 \
      --target_detected_threshold 5 \
      --episodes_file_path VL-LN-Bench/raw_data/mp3d/train/train_iion.json.gz \
      --habitat_config_path dialog_generation/config/tasks/dialog_mp3d.yaml \
      --baseline_config_path dialog_generation/config/expertiments/gen_videos.yaml \
      --normal_category_path dialog_generation/normal_category.json \
      --pointnav_policy_path VL-LN-Bench/pointnav_weights.pth\
      --scene_summary_path VL-LN-Bench/raw_data/mp3d/scene_summary\
      --output_dir <PATH_TO_YOUR_OUTPUT_DIR> \
  ```

### 4. Guidance for Training and Evaluation
- Prerequisites
  ```bash
  # Switch to the dev branch
  cd InternNav
  git checkout dev
  # Link VL-LN Bench data into InternNav
  mkdir projects && cd projects
  ln -s /path/to/your/VL-LN-Bench ./VL-LN-Bench
  ```
  - Write Your Api Key of OpenAI in api_key.txt. 
  ```bash
  # Your final repo structure may look like
  InternNav
  ├── assets/
  ├── internnav/
  │   ├── habitat_vlln_extensions
  │   │   ├── simple_npc
  │   │   │   ├── api_key.txt
  │   ... ... ...
  ...
  ├── projects
  │   ├── VL-LN-Bench/
  │   │   ├── base_model/ 
  │   │   ├── raw_data/ 
  │   │   ├── scene_datasets/
  │   │   ├── traj_data/
  ... ...
  ```
- Start Training
  ```bash
  # Before running, please open this script and make sure 
  # the "llm" path points to the correct checkpoint on your machine.
  sh ./scripts/train/qwenvl_train/train_system2_vlln.sh
  ```
- Start Evaluation
  ```bash
  # If having slurm
  sh ./scripts/eval/bash/srun_eval_dialog.sh
  
  # Or directly run
  python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_dialog_cfg.py
  ```

## 🔗 Citation
If you find our work helpful, please cite:

```bibtex
@misc{huang2025vllnbenchlonghorizongoaloriented,
      title={VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs}, 
      author={Wensi Huang and Shaohao Zhu and Meng Wei and Jinming Xu and Xihui Liu and Hanqing Wang and Tai Wang and Feng Zhao and Jiangmiao Pang},
      year={2025},
      eprint={2512.22342},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.22342}, 
}
```


If you use the specific pretrained models and benchmarks, please kindly cite the original papers involved in our work. Related BibTex entries of our papers are provided below.

<details><summary>Related Work BibTex</summary>

```BibTex
@misc{internvla-n1,
    title = {{InternVLA-N1: An} Open Dual-System Navigation Foundation Model with Learned Latent Plans},
    author = {InternNav Team},
    year = {2025},
    booktitle={arXiv},
}
@inproceedings{vlnpe,
  title={Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities},
  author={Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
@misc{streamvln,
    title = {StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
    author = {Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and Liu, Xihui and Pang, Jiangmiao},
    booktitle={arXiv},
    year = {2025}
}
@misc{navdp,
    title = {NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
    author = {Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang and Jiangmiao Pang},
    year = {2025},
    booktitle={arXiv},
}
```

</details>


## 📄 License

VL-LN's codes are [MIT licensed](LICENSE).
The open-sourced VL-LN data are under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Other datasets, like InternData-N1, inherit their own distribution licenses.

## 👏 Acknowledgement
- [InternNav](https://github.com/InternRobotics/InternNav): InternNav is an All-in-one open-source toolbox for embodied navigation based on PyTorch, Habitat and Isaac Sim.
- [MMScan](https://tai-wang.github.io/mmscan/): MMScan provides a multi-modal 3D scene dataset with hierarchical grounded language annotations, covering holistic aspects on both object- and region-level.
- [VLFM](https://github.com/bdaiinstitute/vlfm): VLFM (Vision-Language Frontier Maps) is a zero-shot semantic navigation method that builds frontier-based occupancy maps from depth and uses a pre-trained vision–language model to produce a language-grounded value map, guiding the agent to explore the most promising frontiers to find unseen target objects in novel environments.
