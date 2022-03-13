#!/usr/bin/env bash

# spellmatch register interactive source_masks target_masks --source-images source_img --target-images target_img initial_assignments initial_transforms
spellmatch register intensities source_img target_img --source-channel "Histone H3" --target-channel "Histone H3" --initial-transforms initial_transforms initial_transforms_refined

spellmatch match source_masks target_masks --algorithm icp --algorithm-args "max_dist=15,min_change=1.0e-9" --prior-transforms initial_transforms_refined scores_icp_forward
spellmatch match source_masks target_masks --algorithm icp --algorithm-args "max_dist=15,min_change=1.0e-9" --prior-transforms initial_transforms_refined --reverse scores_icp_reverse

spellmatch assign scores_icp_forward --strategy forward_max --score-thres 0 --validate initial_assignments assignments_icp_forward
spellmatch assign scores_icp_reverse --strategy reverse_max --score-thres 0 --validate assignments_icp_forward assignments_icp_reverse
spellmatch combine assignments_icp_forward assignments_icp_reverse --strategy union --validate initial_assignments assignments_icp
