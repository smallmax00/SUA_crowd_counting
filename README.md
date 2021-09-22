# Spatial Uncertainty-Aware-Semi-Supervised-Crowd-Counting
ICCV2021 'Spatial Uncertainty-Aware Semi-Supervised Crowd Counting'
---------------------------
Dependencies: requirements.txt

Training code: under prepared, you can check https://github.com/yulequan/UA-MT, as some of our model's structure are built based on it. Thanks Dr.Leyuan for such a wonderful project.

------------SHA-----------
--------------------------

--Download chekpoint_best.pth and put into ./checkpoints/SHA

https://drive.google.com/file/d/1UgCasGAr0SqX8OIVL-vw4EEvHoCg1yHk/view?usp=sharing

Prepare the Test data , then put them into ./Data_Crowd_Counting/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data


--Run the test_SHA.py

The unlabeled data index of SHA train data are in unlabeled_images_index.txt


--------------JHU-----------------
---------------------------------

--Download chekpoint_best.pth and put into ./checkpoints/JHU

https://drive.google.com/file/d/1aWX2s64dSDRkj-oMxqe3tepYzDhW5rNL/view?usp=sharing


--Prepare the Test data , then put them into ./Data_Crowd_Counting/JHU/test

--Run the test_JHU.py

The unlabeled data index of JHU train data are in unlabeled_images_index_JHU.txt


# Citation
If you find our work useful or our work gives you any insights, please cite:
```
@inproceedings{meng2021spatial,
  title = {Spatial Uncertainty-Aware Semi-Supervised Crowd Counting},
  author = {Meng, Yanda and Zhang, Hongrun and Zhao, Yitian and Yang, Xiaoyun and Qian, Xuehsheng and Huang, Xiaowei and Zheng, Yalin}
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year = {2021}
}
```
