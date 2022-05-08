#!/usr/bin/env bash

spellmatch register interactive source_masks target_masks --source-images source_img --target-images target_img initial_assignments initial_transforms
spellmatch register intensities source_img target_img --source-channel "Histone H3" --target-channel "Histone H3" --initial-transforms initial_transforms --show --hold refined_transforms

spellmatch match source_masks target_masks --algorithm icp --algorithm-args "max_dist=15,min_change=1.0e-9" --prior-transforms refined_transforms scores_icp_forward
spellmatch match source_masks target_masks --algorithm icp --algorithm-args "max_dist=15,min_change=1.0e-9" --prior-transforms refined_transforms --reverse scores_icp_reverse
spellmatch assign scores_icp_forward --reverse-scores scores_icp_reverse --max --direction intersect --validate initial_assignments --source-masks source_masks --target-masks target_masks --show 100 assignments_icp

spellmatch match source_masks target_masks --algorithm rigid_cpd --algorithm-args "max_dist=15,update_scale=no" --prior-transforms refined_transforms scores_rigid_cpd_forward
spellmatch match source_masks target_masks --algorithm rigid_cpd --algorithm-args "max_dist=15,update_scale=no" --prior-transforms refined_transforms --reverse scores_rigid_cpd_reverse
spellmatch assign scores_rigid_cpd_forward --reverse-scores scores_rigid_cpd_reverse --max --direction intersect --validate initial_assignments --source-masks source_masks --target-masks target_masks --show 100 assignments_rigid_cpd

spellmatch match source_masks target_masks --algorithm spellmatch --algorithm-args "intensity_transform=numpy.arcsinh,scores_tol=2.0e-4,spatial_cdist_prior_thres=15,max_spatial_cdist=25,degree_weight=1,intensity_weight=1,intensity_interp_lmd=0.5,intensity_shared_pca_n_components=3,intensity_all_cca_fit_k_closest=200,intensity_all_cca_fit_k_most_certain=100,intensity_all_cca_n_components=3,distance_weight=1" --source-images source_img --target-images target_img --prior-transforms refined_transforms scores_spellmatch
spellmatch assign scores_spellmatch --max --direction intersect --validate initial_assignments --source-masks source_masks --target-masks target_masks --show 100 assignments_spellmatch
