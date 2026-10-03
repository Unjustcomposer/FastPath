import numpy as np

def compute_froc(lesion_preds, ground_truth_lesions, evaluation_mask=None, thresholds_count=50):
    """
    Computes the Free-Response ROC curve and FROC score for CAMELYON16 evaluation.
    
    Args:
        lesion_preds: Dictionary mapping WSI name to list of dicts with 'point': (x,y) and 'probability': float
        ground_truth_lesions: Dictionary mapping WSI name to list of ground truth lesion polygons or points
        evaluation_mask: Optional mask dictionary mapping WSI name to an evaluation region mask (not strictly needed if preds are already masked)
        thresholds_count: Number of thresholds to evaluate
        
    Returns:
        froc_score: Average sensitivity at the predefined FP rates
        froc_curve: Tuple of (fps_per_wsi, sensitivities)
    """
    predefined_fp_rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    all_probs = []
    
    # Collect all probabilities to define thresholds
    for wsi_name, preds in lesion_preds.items():
        all_probs.extend([p['probability'] for p in preds])
        
    if not all_probs:
        return 0.0, (predefined_fp_rates, [0.0]*len(predefined_fp_rates))
        
    thresholds = np.linspace(min(all_probs), max(all_probs), thresholds_count)
    
    # Pre-calculate distances or hit-testing for efficiency
    # Assuming ground_truth_lesions contains list of dicts with 'center' and 'radius' for simplicity,
    # or bounding boxes. A robust implementation would use shapely polygons.
    # We will simulate the CAMELYON logic.
    
    wsi_names = list(ground_truth_lesions.keys())
    num_wsis = len(wsi_names)
    total_gt_lesions = sum(len(lesions) for lesions in ground_truth_lesions.values())
    
    if total_gt_lesions == 0:
        return 0.0, (predefined_fp_rates, [0.0]*len(predefined_fp_rates))
        
    sensitivities = []
    fps_per_image = []
    
    for thresh in sorted(thresholds, reverse=True):
        tp = 0
        fp = 0
        
        for wsi_name in wsi_names:
            preds = lesion_preds.get(wsi_name, [])
            gts = ground_truth_lesions.get(wsi_name, [])
            
            # Filter by threshold
            valid_preds = [p for p in preds if p['probability'] >= thresh]
            
            # Match predictions to GT lesions
            # A GT lesion is considered detected if ANY valid prediction falls within it
            # False positives are predictions that don't fall within ANY GT lesion
            # Note: For robust CAMELYON16, we usually check if distance from pred point to GT center is within radius
            # or if point is inside Polygon.
            
            gt_detected = [False] * len(gts)
            
            for pred in valid_preds:
                px, py = pred['point']
                is_hit = False
                
                for idx, gt in enumerate(gts):
                    # Simplified hit check (assuming 'center' and 'radius' are available)
                    if 'center' in gt and 'radius' in gt:
                        cx, cy = gt['center']
                        if (px - cx)**2 + (py - cy)**2 <= gt['radius']**2:
                            is_hit = True
                            gt_detected[idx] = True
                            break
                    # Mock logic for shapely polygons
                    elif 'polygon' in gt:
                        # e.g., if gt['polygon'].contains(Point(px, py))
                        pass
                        
                if not is_hit:
                    fp += 1
                    
        tp = sum(gt_detected)
        
        sensitivity = tp / total_gt_lesions
        fp_per_wsi = fp / num_wsis if num_wsis > 0 else 0.0
        
        sensitivities.append(sensitivity)
        fps_per_image.append(fp_per_wsi)
        
    # Interpolate sensitivities at predefined FP rates
    froc_score = 0.0
    final_sensitivities = []
    
    for rate in predefined_fp_rates:
        # Find sensitivity where FP rate is roughly equal to 'rate'
        # Since fps_per_image is monotonically increasing (as threshold decreases)
        idx = np.searchsorted(fps_per_image, rate)
        if idx == 0:
            sens = sensitivities[0] if fps_per_image else 0.0
        elif idx == len(fps_per_image):
            sens = sensitivities[-1]
        else:
            # Linear interpolation
            fp1, fp2 = fps_per_image[idx-1], fps_per_image[idx]
            s1, s2 = sensitivities[idx-1], sensitivities[idx]
            if fp2 == fp1:
                sens = s1
            else:
                sens = s1 + (s2 - s1) * (rate - fp1) / (fp2 - fp1)
                
        final_sensitivities.append(sens)
        froc_score += sens
        
    froc_score /= len(predefined_fp_rates)
    
    return froc_score, (predefined_fp_rates, final_sensitivities)
