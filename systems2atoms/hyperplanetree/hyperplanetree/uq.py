# Uncertainty quantification strategies largely inspired by:
#
# Cameron J Gruich et al 2023. Mach. Learn.: Sci. Technol. 4 025019
# https://www.doi.org/10.1088/2632-2153/accace
#
# Uncertainty Toolbox
# https://github.com/uncertainty-toolbox/uncertainty-toolbox/

import torch

def calculate_uncertainty_metrics(
        y_pred,
        uncertainty,
        y_true,
        num_bins = 100,
        resolution = 99,
    ):

    try:
        from uncertainty_toolbox import get_all_metrics
    except ImportError:
        raise ImportError('For uncertainty quantification, please install Uncertainty Toolbox: https://uncertainty-toolbox.github.io/')
    
    y_pred = y_pred.flatten()
    uncertainty = uncertainty.flatten()
    y_true = y_true.flatten()

    out = {
        'accuracy': {},
        'sharpness': {},
        'dispersion': {},
        'calibration': {},
        'tightness': {},
    }

    utb = get_all_metrics(
        y_pred=y_pred.numpy(),
        y_true=y_true.numpy(),
        y_std=uncertainty.numpy(),
        verbose = False,
        num_bins = num_bins,
        resolution = resolution,
        )

    out['accuracy'] = utb['accuracy']

    out['sharpness']['rms_unc'] = torch.sqrt(torch.mean(uncertainty**2)).item()
    out['sharpness']['mean_unc'] = torch.mean(uncertainty).item()
    out['sharpness']['median_unc'] = torch.median(uncertainty).item()
    out['sharpness']['75_percentile_unc'] = torch.quantile(uncertainty, 0.75).item()
    out['sharpness']['99_percentile_unc'] = torch.quantile(uncertainty, 0.99).item()

    out['dispersion']['interquartile_range'] = (torch.quantile(uncertainty, 0.75) - torch.quantile(uncertainty, 0.25)).item()
    out['dispersion']['standard_deviation'] = torch.std(uncertainty).item()
    out['dispersion']['coefficient_of_variation'] = (torch.std(uncertainty) / torch.mean(uncertainty)).item()
    out['dispersion']['98_percentile_range'] = (torch.quantile(uncertainty, 0.99) - torch.quantile(uncertainty, 0.01)).item()
    out['dispersion']['median_relative_deviation'] = (torch.median(torch.abs(uncertainty - torch.median(uncertainty))) / torch.median(uncertainty)).item()
    out['dispersion']['mean_relative_deviation'] = (torch.mean(torch.abs(uncertainty - torch.mean(uncertainty))) / torch.mean(uncertainty)).item()

    out['calibration']['miscalibrated_area'] = utb['avg_calibration']['miscal_area']
    out['calibration']['rms_calibration_error'] = utb['avg_calibration']['rms_cal']
    out['calibration']['mean_abs_calibration_error'] = utb['avg_calibration']['ma_cal']

    out['tightness'] = utb['scoring_rule']

    return out 