 TinyFace Face Recognition Benchmark




# ======================== Dataset Structure ========================== #



"Training_Set": 

Ordered by face identities, i.e. each directory contains face images from a specific training identity. There are a total of 2,570 directories each is named by the corresponding identity. The image file name is in format of [PersonID]_[ImageName].jpg.





"Testing_Set":


	- "Gallery_Match": containing 4,443 mated gallery images from 2,569 test identities (IDs).

	- "Gallery_Distractor": containing 153,428 distractor gallery face images without mated probe true match images in the "Probe" folder.

	- "Probe": containing 3,728 probe images from 2,569 test IDs, with mated gallery true match images in the "Gallery_Match" folder.

	- "gallery_match_img_ID_pairs.mat": the mated gallery image names and the corresponding face IDs.

	- "probe_img_ID_pairs.mat": the probe image names and the corresponding face IDs.





# ======================== Evaluation Instruction ========================== #

*** Face Identification Evaluation ("Face_Identification_Evaluation") *** 


1. Extract features of "Gallery_Match" images according to the order defined by "gallery_match_img_ID_pairs.mat", put them in a matrix called "gallery_feature_map" ([image_number]_by_[feature_dimension]), and save the matrix in a mat file named "gallery.mat"



2. Extract features of "Probe" images according to the order defined by "probe_img_ID_pairs.mat", put them in a matrix called "probe_feature_map" ([image_number]_by_[feature_dimension]), and save the matrix in a mat file named "probe.mat"



3. Extract features of "Gallery_Distractor" images, put them in a matrix called "distractor_feature_map" ([image_number]_by_[feature_dimension]), and save the matrix in a mat file named "distractor.mat"



4. Put the above three mat files in the directory named "features"



5. Run "test_face_identification.m" to get the identification performance







# ======================== Citation ===================================== #

@article{cheng2018low,
  title={Low-Resolution Face Recognition},
  author={Cheng, Zhiyi and Zhu, Xiatian and Gong, Shaogang},
  booktitle={Asian Conference on Computer Vision},
  year={2018}
}
