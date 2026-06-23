import numpy as np
import pandas as pd
from scipy import stats


class StatisticalTester:
    def run(self, scoring_df, metric_col='SupportVectorMachine_macro_f1'):
        try:
            import scikit_posthocs as sp
        except ImportError:
            raise ImportError("scikit-posthocs required: pip install scikit-posthocs")

        methods = list(scoring_df['name'].unique())
        k_values = sorted(scoring_df['selected'].unique())

        friedman_rows = []
        for k in k_values:
            sub = scoring_df[scoring_df['selected'] == k]
            score_vectors = []
            valid_methods = []
            min_len = None

            for m in methods:
                s = sub[sub['name'] == m][metric_col].dropna().values
                if len(s) > 0:
                    score_vectors.append(s)
                    valid_methods.append(m)
                    min_len = min(min_len, len(s)) if min_len is not None else len(s)

            if len(score_vectors) < 2 or (min_len is not None and min_len < 2):
                continue

            aligned = [s[:min_len] for s in score_vectors]
            stat, p = stats.friedmanchisquare(*aligned)
            friedman_rows.append({
                'k': k, 'statistic': stat, 'p_value': p,
                'significant': p < 0.05, 'methods': str(valid_methods),
            })

        friedman_df = pd.DataFrame(friedman_rows)

        # Nemenyi post-hoc across all k values — rows = (dataset × k), columns = methods
        dataset_col = 'dataset' if 'dataset' in scoring_df.columns else None
        index_cols = [dataset_col, 'selected'] if dataset_col else ['selected']
        pivot = scoring_df.pivot_table(
            index=index_cols, columns='name', values=metric_col, aggfunc='mean'
        ).dropna()

        if len(pivot) >= 3 and pivot.shape[1] >= 2:
            nemenyi_result = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi_result.columns = pivot.columns
            nemenyi_result.index = pivot.columns
        else:
            nemenyi_result = pd.DataFrame()

        return friedman_df, nemenyi_result

    def wilcoxon_pairwise(self, scoring_df, metric_col='SupportVectorMachine_macro_f1'):
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            raise ImportError("statsmodels required: pip install statsmodels")

        methods = list(scoring_df['name'].unique())
        pairs = []

        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                a = scoring_df[scoring_df['name'] == m1][metric_col].dropna().values
                b = scoring_df[scoring_df['name'] == m2][metric_col].dropna().values
                min_len = min(len(a), len(b))
                if min_len < 2:
                    continue
                a, b = a[:min_len], b[:min_len]
                stat, p = stats.wilcoxon(a, b)
                pairs.append({'method_a': m1, 'method_b': m2, 'statistic': stat, 'p_value': p})

        if not pairs:
            return pd.DataFrame(columns=['method_a', 'method_b', 'statistic', 'p_value',
                                         'p_corrected', 'significant'])

        result_df = pd.DataFrame(pairs)
        _, p_corr, _, _ = multipletests(result_df['p_value'].values, method='bonferroni')
        result_df['p_corrected'] = p_corr
        result_df['significant'] = p_corr < 0.05
        return result_df
