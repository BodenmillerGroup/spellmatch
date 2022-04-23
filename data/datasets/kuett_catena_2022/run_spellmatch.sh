#!/usr/bin/env bash

spellmatch register interactive source_masks target_masks --source-images source_img --target-images target_img initial_assignments initial_transforms
spellmatch register intensities source_img target_img --source-channel "Histone H3" --target-channel "Histone H3" --initial-transforms initial_transforms --show --hold refined_transforms

spellmatch match source_masks target_masks --algorithm icp --algorithm-args "max_dist=15,min_change=1.0e-9" --prior-transforms refined_transforms scores_icp_forward
spellmatch match source_masks target_masks --algorithm icp --algorithm-args "max_dist=15,min_change=1.0e-9" --prior-transforms refined_transforms --reverse scores_icp_reverse
spellmatch assign scores_icp_forward --reverse-scores scores_icp_reverse --max-only --direction union --validate initial_assignments --source-masks source_masks --target-masks target_masks --show 100 assignments_icp

# spellmatch match source_masks target_masks --algorithm spellmatch --algorithm-args "transform_tol=1.0e-3,intensity_transform=numpy.log1p,adj_radius=15.0,degree_weight=0.0,intensity_weight=1.0,celldist_weight=0.0,spatial_cdist_prior_thres=15.0,max_spatial_cdist=15.0" --source-images source_img --target-images target_img --prior-transforms refined_transforms scores_spellmatch_forward
# spellmatch match source_masks target_masks --algorithm spellmatch --algorithm-args "transform_tol=1.0e-3,intensity_transform=numpy.log1p,adj_radius=15.0,degree_weight=0.0,intensity_weight=1.0,celldist_weight=0.0,spatial_cdist_prior_thres=15.0,max_spatial_cdist=15.0" --source-images source_img --target-images target_img --prior-transforms refined_transforms --reverse scores_spellmatch_reverse
# spellmatch assign scores_spellmatch_forward --strategy forward_max --score-thres 0 --validate initial_assignments --source-masks source_masks --target-masks target_masks --show 100 assignments_spellmatch_forward
# spellmatch assign scores_spellmatch_reverse --strategy reverse_max --score-thres 0 --validate assignments_spellmatch_forward --source-masks source_masks --target-masks target_masks --show 100 assignments_spellmatch_reverse
# spellmatch combine assignments_spellmatch_forward assignments_spellmatch_reverse --strategy union --validate initial_assignments --source-masks source_masks --target-masks target_masks --show 100 assignments_spellmatch
