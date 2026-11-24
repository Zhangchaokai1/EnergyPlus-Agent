"""Descriptive SHAP directions; never infer a slope from mean(phi).

The |r| >= 0.5 cutoff is a conservative display rule, not a significance test
or evidence of global monotonicity/causality. A directional label also requires
opposite signs of mean SHAP in the low/high feature quartiles, with at least
75% of the high-value quartile sharing the stated sign. Otherwise the
tool reports a mixed/weak pattern and leaves the signed plot authoritative.
"""
import math

DIRECTION_CORRELATION_MIN = 0.5

def _finite(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None

def direction_details(row, language='en'):
    corr = _finite(row, 'corr_feature_contribution')
    low = _finite(row, 'low_quartile_mean_shap')
    high = _finite(row, 'high_quartile_mean_shap')
    neg_fraction = _finite(row, 'high_quartile_negative_fraction')
    pos_fraction = _finite(row, 'high_quartile_positive_fraction')
    trend = 'mixed'
    if None not in (corr, low, high):
        if corr <= -DIRECTION_CORRELATION_MIN and low > 0 > high and neg_fraction is not None and neg_fraction >= 0.75:
            trend = 'negative'
        elif corr >= DIRECTION_CORRELATION_MIN and low < 0 < high and pos_fraction is not None and pos_fraction >= 0.75:
            trend = 'positive'
    texts = {
        'en': {
            'negative': 'High values generally have negative SHAP contributions (lower than reference)',
            'positive': 'High values generally have positive SHAP contributions (higher than reference)',
            'mixed': 'Mixed or weak directional pattern; inspect the signed SHAP distribution',
        },
        'zh': {
            'negative': '高取值通常对应负 SHAP 贡献（相对参考点降低指标）',
            'positive': '高取值通常对应正 SHAP 贡献（相对参考点提高指标）',
            'mixed': '方向混合或关联较弱，需结合有符号 SHAP 分布判断',
        },
    }
    return {'direction': texts[language][trend], 'directionCode': trend,
            'featureShapCorrelation': corr, 'lowQuartileMeanShap': low,
            'highQuartileMeanShap': high, 'highQuartileNegativeFraction': neg_fraction,
            'highQuartilePositiveFraction': pos_fraction}
